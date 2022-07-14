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

QUEUE_SIZE=16
BATCH_SIZE=16

port = "5555"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(f"tcp://*:{port}")

server_port = "5556"
pair_socket = context.socket(zmq.PAIR)
pair_socket.bind(f"tcp://*:{server_port}")

# start deepspeed server
# subprocess.Popen([sys.executable,"-u", "-m", "deepspeed.launcher.runner", "--num_gpus", "1", "scripts/inference/bloom-ds-inference-server.py", "--name", "bigscience/bigscience-small-testing"])
subprocess.Popen([sys.executable,"-u", "-m", "deepspeed.launcher.runner", "--num_gpus", "16", "scripts/inference/bloom-ds-inference-server.py", "--name", "bigscience/bloom"])


client_conntected = False
while not client_conntected:
    print("Waiting for clients to connect")
    response = pair_socket.recv()
    if response.decode() == "READY":
        client_conntected = True
        print("clients connected")

def server_loop(q):

    while True:
        items = [q.get()]
        while len(items) < BATCH_SIZE:
            try:
                item = q.get(False)
            except Empty:
                break
            items.append(item)

        all_inputs = [item[0] for item in items]
        all_queues = [item[-1] for item in items]

        socket.send_pyobj(all_inputs)
        out = pair_socket.recv_pyobj()

        for string, response_queue in zip(out, all_queues):
            response_queue.put(string)

def run_app(q):
    app = Flask(__name__)

    @app.route("/generate", methods=["POST"])
    def generate():
        body = request.json

        qsize = q.qsize()
        print("Queue size", qsize)

        if qsize >= QUEUE_SIZE:
            return make_response({"error": "Queue full , try again later"}, 503)

        response_queue = Queue()
        q.put((body.get("inputs", "Hello"), response_queue))

        out = response_queue.get()

        return make_response(jsonify([{"generated_text": out}]), 200)

    app.run(port=8000, host="127.0.0.1")


if __name__ == "__main__":
    q = Queue()
    threading.Thread(target=run_app, args=(q,)).start()

    server_loop(q)

