import requests
import subprocess
import time
import signal
import threading
from flask import Flask, Response, jsonify
import re  # Regular expression to extract the last floating-point value
import numpy as np

app = Flask(__name__)
metrics_data = {}


def extract_last_float_value(line):
    # Using regex to extract the last floating-point number from a line
    match = re.search(r"(\d+\.\d+|\d+)(?:\s|$)", line)  # Match both integers and floats
    return float(match.group(1)) if match else None


def get_metrics():
    try:
        response = requests.get("http://0.0.0.0:9904/metrics", timeout=5)
        response.raise_for_status()
        lines = response.text.split("\n")

        # Create a dictionary to store metrics
        metrics = {
            "node_used_cpu": [],
            "node_used_gpu": [],
            "node_used_mem": [],
        }

        for line in lines:
            # For node metrics
            if "node_used_cpu" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["node_used_cpu"].append(value)
            elif "node_used_gpu" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["node_used_gpu"].append(value)
            elif "node_used_mem" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["node_used_mem"].append(value)

        return metrics
    except requests.RequestException:
        return {}


def find_ports():
    try:
        result = subprocess.run(["ss", "-nlpt"], capture_output=True, text=True, check=True)
        lines = result.stdout.split("\n")
        ports = []
        for line in lines:
            if "0.0.0.0:11" in line:  # Modify to find any port starting with "11"
                parts = line.split()
                for part in parts:
                    if part.startswith("0.0.0.0:11"):  # Capture all ports starting with 11
                        port = part.split(":")[-1]
                        ports.append(port)
        return list(set(ports))
    except subprocess.CalledProcessError:
        return []


def get_port_metrics(port):
    try:
        response = requests.get(f"http://0.0.0.0:{port}/metrics", timeout=5)
        response.raise_for_status()
        lines = response.text.split("\n")
        
        # Create a dictionary to store container and job training metrics for each port
        metrics = {
            "container_used_cpu": [],
            "container_used_gpu": [],
            "container_used_mem": [],
            "job_training_loss": [],
            "job_training_accuracy": []
        }

        for line in lines:
            # For container metrics
            if "container_used_cpu" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["container_used_cpu"].append(value)
            elif "container_used_gpu" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["container_used_gpu"].append(value)
            elif "container_used_mem" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["container_used_mem"].append(value)

            # For job training metrics
            elif "job_training_loss" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_training_loss"].append(value)
            elif "job_training_accuracy" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_training_accuracy"].append(value)

        return metrics
    except requests.RequestException:
        return {}


def update_metrics():
    global metrics_data
    while True:
        node_metrics = get_metrics()
        ports = find_ports()  # Find all ports starting with "11"
        port_metrics = {
            "container_used_cpu": [],
            "container_used_gpu": [],
            "container_used_mem": [],
            "job_training_loss": [],
            "job_training_accuracy": []
        }

        # Collect metrics from each of the found ports
        for port in ports:
            port_metrics_for_this_port = get_port_metrics(port)
            for metric, values in port_metrics_for_this_port.items():
                port_metrics[metric].extend(values)  # Accumulate values for each metric

        # Combine node and port metrics in the desired structure
        metrics_data = {**node_metrics, **port_metrics}
        time.sleep(3)

def flatten_metrics(metrics):
    return np.concatenate([np.array(v, dtype=np.float32) for v in metrics.values()])

@app.route("/state", methods=["GET"])
def serve_metrics():
    flattened_state = flatten_metrics(metrics_data)
    return jsonify(flattened_state.tolist())


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

