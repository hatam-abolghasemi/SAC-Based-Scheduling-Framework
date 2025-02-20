import requests
import subprocess
import time
import signal
import threading
from flask import Flask, Response


app = Flask(__name__)
metrics_data = ""


def get_metrics():
    try:
        response = requests.get("http://0.0.0.0:9904/metrics", timeout=5)
        response.raise_for_status()
        lines = [line for line in response.text.split("\n") if line.startswith("node_used")]
        return lines
    except requests.RequestException:
        return []


def find_ports():
    try:
        result = subprocess.run(["ss", "-nlpt"], capture_output=True, text=True, check=True)
        lines = result.stdout.split("\n")
        ports = []
        for line in lines:
            if "0.0.0.0:11" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("0.0.0.0:11"):
                        port = part.split(":")[-1]
                        ports.append(port)
        return list(set(ports))
    except subprocess.CalledProcessError:
        return []


def get_port_metrics(port):
    try:
        response = requests.get(f"http://0.0.0.0:{port}/metrics", timeout=5)
        response.raise_for_status()
        lines = [line for line in response.text.split("\n") if line.startswith("job_training") or line.startswith("container_used")]
        return lines
    except requests.RequestException:
        return []


def update_metrics():
    global metrics_data
    while True:
        node_metrics = get_metrics()
        ports = find_ports()
        port_metrics = []
        for port in ports:
            port_metrics.extend(get_port_metrics(port))
        metrics_data = "\n".join(node_metrics + port_metrics)
        time.sleep(3)


@app.route("/state", methods=["GET"])
def serve_metrics():
    return Response(metrics_data, mimetype="text/plain")


def run_server():
    app.run(host="0.0.0.0", port=9907, debug=False, use_reloader=False)


def signal_handler(sig, frame):
    print("Shutting down...")
    exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    updater_thread = threading.Thread(target=update_metrics, daemon=True)
    updater_thread.start()
    run_server()

