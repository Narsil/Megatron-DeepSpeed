from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import threading
from queue import Queue, Empty
import time
import zmq
import uvicorn
import subprocess
import sys
from flask import Flask, jsonify, make_response, request

QUEUE_SIZE=32
BATCH_SIZE=32

port = "5555"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(f"tcp://*:{port}")

server_port = "5556"
pair_socket = context.socket(zmq.PAIR)
pair_socket.bind(f"tcp://*:{server_port}")

# start deepspeed server
# subprocess.Popen([sys.executable,"-u", "-m", "deepspeed.launcher.runner", "--num_gpus", "4", "ds-server3.py", "--name", "bigscience/bigscience-small-testing"])
subprocess.Popen([sys.executable,"-u", "-m", "deepspeed.launcher.runner", "--num_gpus", "16", "ds-server3.py", "--name", "bigscience/bloom"])
#subprocess.Popen([sys.executable,"-u", "-m", "deepspeed.launcher.runner", "--num_gpus", "16", "ds-server3.py", "--name", "bigscience/bloom-1b3"])


client_conntected = False
while not client_conntected:
    print("Waiting for clients to connect")
    response = pair_socket.recv()
    if response.decode() == "READY":
        client_conntected = True
        print("clients connected")


def run_app(q):
    app = Flask(__name__)

    @app.route("/generate", methods=["POST"])
    def generate():
        body = request.json

        qsize = q.qsize()
        print("Queue size", qsize)

        if qsize >= QUEUE_SIZE:
            return make_response({"error": "Queue full , try again later"}, 503)
        if "inputs" not in body:
            return make_response({"error": "`inputs` is required"}, 400)

        inputs = body.get("inputs", "Hello")
        parameters = body.get("parameters", {})

        if parameters.get("max_new_tokens", 20) > 512:
            return make_response({"error": "You cannot generate more than 100 new tokens, at least for now"}, 400)

        if len(inputs) > 2000:
            return make_response({"error": "This prompt is very long, we're temporarily disabling these"}, 400)

        # Remove seed we can't use it in a group.
        parameters.pop("seed", None)

        response_queue = Queue()
        q.put((inputs, parameters, response_queue))

        out = response_queue.get()

        return make_response(jsonify([{"generated_text": out}]), 200)

    app.run(port=8000, host="127.0.0.1")


def server_loop(q):
    
    remaining_items = []

    while True:
        print("Server loop")
        last_parameters = remaining_items[0][1] if remaining_items else None

        items = [remaining_items.pop()] if remaining_items else []
        i = 0
        while i < len(remaining_items):
            parameters = remaining_items[i][1]
            if last_parameters is not None and parameters != last_parameters:
                items.append(remaining_items.pop(i))
            else:
                i += 1

        while len(items) < BATCH_SIZE:
            if len(items) > 0:
                try:
                    item = q.get(False)
                except Empty:
                    break
            else:
                item = q.get()

            (input_text, parameters, response_queue) = item

            if last_parameters is not None and parameters != last_parameters:
                print(f"Ignoring new parameters {parameters}")
                remaining_items.append(item)
                continue

            items.append(item)
            last_parameters = parameters

        print(f"Found {len(items)} items")
        all_inputs = [item[0] for item in items]
        all_queues = [item[-1] for item in items]

        print(f"[loop] Sending generation of batch size {len(all_inputs)} with {last_parameters}")
        socket.send_pyobj((all_inputs, last_parameters))
        print(f"[loop] Receiving")
        out = pair_socket.recv_pyobj()
        print(f"[loop] Receveived loop")

        for string, response_queue in zip(out, all_queues):
            response_queue.put(string)
            print("---")
            print(f"Sent back {string}" )
            print("---")
if __name__ == "__main__":
    q = Queue()
    threading.Thread(target=run_app, args=(q,)).start()

    server_loop(q)

