# usage:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom
#
# to run benchmarks:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom --benchmark
#


# This is going to improve, but at the moment, the process is a bit cumbersome - we first use
# 1. use Deepspeed-ZeRO to instantiate the model on GPUs, w/o loading the checkpoints,
# 2. free the allocated storage
# 3. start Deepspeed-Inference and only now load the checkpoint
# 4. run generate
# Done.
#


import glob
import datetime
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
import deepspeed
import io
import math
import zmq
import sys
import json
import os
import gc
import torch
import torch.distributed as dist
import time

t_start = time.time()

num_tokens = 100

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
args = parser.parse_args()


port = "5555"
# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(f"tcp://localhost:{port}")
socket.subscribe(b"")
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

deepspeed.init_distributed('nccl')
rank = dist.get_rank()


# reproducible randomization / seed setting
# ----------------------------------- #
import random, torch, numpy as np
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    random.seed(seed)    # python RNG
    np.random.seed(seed) # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)          # cpu + cuda
    torch.cuda.manual_seed_all(seed) # multi-gpu
    if use_seed: # slower speed! https://pytorch.org/docs/stable/notes/randomness.html#cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    return seed

# no longer needed with fixed branch
#enforce_reproducibility(1)


### Model loading and instantiating on GPU (via ZeRO)

def get_checkpoint_files(pretrained_model_name_or_path):
    # XXX: I just hacked this one together to automatically handle the fetching of the model file or
    # shards into cache and returning the cached entries - note that I removed most arguments

    from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, cached_path, hf_bucket_url, is_offline_mode
    from transformers.utils.hub import EntryNotFoundError
    from transformers.modeling_utils import get_checkpoint_shard_files

    cache_dir = None
    is_sharded = False

    # XXX: preparation for revision branches if needed
    revision = None
    #revision = "sharded"

    # this supports nodes with no network (so you need to pre-cache the model and the tokenizer with
    # python -c "from transformers import AutoModel; AutoModel.from_pretrained('bigscience/bloom')"
    if is_offline_mode():
        print("Offline mode: forcing local_files_only=True")
        local_files_only = True
    else:
        local_files_only = False

    filename = WEIGHTS_NAME
    archive_file = hf_bucket_url(pretrained_model_name_or_path, filename=filename, revision=revision)

    try:
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, local_files_only=local_files_only,)
        return [resolved_archive_file]

    except (EntryNotFoundError, FileNotFoundError):
        if filename == WEIGHTS_NAME:
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=WEIGHTS_INDEX_NAME,
                revision=revision,
            )
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            is_sharded = True

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            revision=revision
        )

        return resolved_archive_file

model_name = args.name

#print(get_checkpoint_files(model_name))

if rank == 0:
    print(f"*** Loading the model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# XXX: can't automatically derive dtype via config's `from_pretrained`
#dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16


# use one of these args to `init_inference`
# 1. injection_policy is the slower version, but it's plain pytorch so it'll always work
# 2. replace_with_kernel_inject is the faster one (fast fused kernels)
kernel_inject = True
#kernel_inject = False

if kernel_inject:
    # XXX: for now ds-inference only works with fp16
    dtype = torch.float16
else:
    dtype = torch.bfloat16

# Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
with deepspeed.OnDevice(dtype=dtype, device='meta'):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

if args.benchmark:
    deepspeed.runtime.utils.see_memory_usage('post-from-pretrained', force=True)

model = model.eval()


if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('post-init-ds-zero-init', force=True)


# local tp shards
LOAD_TP = False
if LOAD_TP:
    checkpoint_type = "tp"
    checkpoint_dir = "/home/nicolas_huggingface_co/src/Megatron-DeepSpeed/bloom-tp"
    checkpoint_files = glob.glob(f"{checkpoint_dir}/*pt")
else:
    # hf checkpoint
    checkpoint_files = get_checkpoint_files(model_name)
    checkpoint_type = "pp" # normal hf hub checkpoint

if rank == 0:
    print("Checkpoint files:", checkpoint_files)
    print("Checkpoint type:", checkpoint_type)

checkpoints_json = "checkpoints.json"
def write_checkponts_json():

    with io.open(checkpoints_json, 'w', encoding='utf-8') as f:

        data = {
            "type": "BLOOM-176B",
            "checkpoints": checkpoint_files,
            "version": 1.0,
            "parallelization": checkpoint_type,
        }
#        if checkpoint_type is not None:
#            data["parallelization"] = checkpoint_type

        json.dump(data, f)

if rank == 0:
    write_checkponts_json()
dist.barrier()

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('pre-ds-inference-init', force=True)

if kernel_inject:
    kwargs = dict(replace_with_kernel_inject=True)
else:
    kwargs = dict(injection_policy={BloomBlock: ('self_attention.dense', 'mlp.dense_4h_to_h')})

# kwargs["save_mp_checkpoint_path"] = checkpoint_dir

print(checkpoints_json)

#checkpoints_json=None
model = deepspeed.init_inference(model,
                                 mp_size=world_size,
                                 dtype=torch.half,
                                 checkpoint=checkpoints_json,
                                 **kwargs,
                                 )

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage('post-ds-inference-init', force=True)


model = model.module

if args.benchmark:
    t_ready = time.time()


### Generate

if rank == 0:
    print(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

# generate_kwargs = dict(min_length=num_tokens, max_length=num_tokens, do_sample=False)
# generate_kwargs = dict(min_length=num_tokens, max_length=num_tokens, do_sample=True)
# if rank == 0:
#     print(f"Generate args {generate_kwargs}")

def generate(inputs, generate_kwargs):
    """ returns a list of pairs of inputs and outputs """

    tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)

    for t in tokens:
        if torch.is_tensor(tokens[t]):
            tokens[t] = tokens[t].to(torch.cuda.current_device())

    greedy_output = model.generate(**tokens, **generate_kwargs,synced_gpus=True)

    outputs = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

    return outputs


if local_rank == 0:
    pair_port = "5556"
    pair_socket = context.socket(zmq.PAIR)
    pair_socket.connect(f"tcp://localhost:{pair_port}")
    pair_socket.send(b"READY")


def predict(body):
    # pop inputs for pipeline
    inputs, parameters = body
    prediction = generate(inputs, parameters)
    return prediction

# Process 5 updates
while True:
    # print(f"[{datetime.datetime.now()}] [DS {rank}] Receiving")
    body = socket.recv_pyobj()
    # print(f"[{datetime.datetime.now()}] [DS {rank}] Predicting {body}")
    pred = predict(body)
    # print(f"[{datetime.datetime.now()}] [DS {rank}] Predicted {body}")
    if local_rank == 0:
        # print(f"[{datetime.datetime.now()}] [DS {rank}] Sending back {body}")
        pair_socket.send_pyobj(pred)
        # print(f"[{datetime.datetime.now()}] [DS {rank}] Sent back {body}")


