import requests
import subprocess
import time
import signal
import threading
from flask import Flask, Response, jsonify
import re
import numpy as np

app = Flask(__name__)
metrics_data = {}

def extract_last_float_value(line):
    match = re.search(r"(\d+\.\d+|\d+)(?:\s|$)", line)
    return float(match.group(1)) if match else None


def get_metrics():
    try:
        response = requests.get("http://0.0.0.0:9804/metrics", timeout=5)
        response.raise_for_status()
        lines = response.text.split("\n")
        metrics = {
            "node_cpu_utilization": [],
            "node_gpu_utilization": [],
            "node_mem_utilization": [],
        }
        for line in lines:
            if "node_cpu_utilization" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["node_cpu_utilization"].append(value)
            elif "node_gpu_utilization" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["node_gpu_utilization"].append(value)
            elif "node_mem_utilization" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["node_mem_utilization"].append(value)
        return metrics
    except requests.RequestException:
        return {}

def find_ports():
    try:
        result = subprocess.run(["ss", "-nlpt"], capture_output=True, text=True, check=True)
        lines = result.stdout.split("\n")
        ports = set()
        for line in lines:
            match = re.search(r":(\d+)", line)
            if match:
                port = int(match.group(1))
                if port > 11000:
                    ports.add(str(port))
        return sorted(ports)
    except subprocess.CalledProcessError:
        return []

def get_port_metrics(port):
    try:
        response = requests.get(f"http://0.0.0.0:{port}/metrics", timeout=5)
        response.raise_for_status()
        lines = response.text.split("\n")
        metrics = {
            "container_used_cpu": [],
            "container_used_gpu": [],
            "container_used_mem": [],
            "job_training_loss": [],
            "job_training_accuracy": [],
            "job_training_progress": [],
            "job_schedule_moment": [],
            "job_generation_moment": [],
            "job_elapsed_time": [],
            "job_required_epoch": [],
            "job_passed_epoch": [],
            "job_model_complexity": [],
            "job_dataset_complexity": [],
            "job_batch_size": [],
            "job_learning_rate": [],
        }
        for line in lines:
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
            elif "job_training_loss" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_training_loss"].append(value)
            elif "job_training_accuracy" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_training_accuracy"].append(value)
            elif "job_training_progress" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_training_progress"].append(value)
            elif "job_schedule_moment" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_schedule_moment"].append(value)
            elif "job_generation_moment" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_generation_moment"].append(value)
            elif "job_elapsed_time" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_elapsed_time"].append(value)
            elif "job_required_epoch" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_required_epoch"].append(value)
            elif "job_passed_epoch" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_passed_epoch"].append(value)
            elif "job_model_complexity" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_model_complexity"].append(value)
            elif "job_dataset_complexity" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_dataset_complexity"].append(value)
            elif "job_batch_size" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_batch_size"].append(value)
            elif "job_learning_rate" in line:
                value = extract_last_float_value(line)
                if value is not None:
                    metrics["job_learning_rate"].append(value)
        return metrics
    except requests.RequestException:
        return {}


def update_metrics():
    global metrics_data
    while True:
        node_metrics = get_metrics()
        ports = find_ports()
        port_metrics = {
            "container_used_cpu": [],
            "container_used_gpu": [],
            "container_used_mem": [],
            "job_training_loss": [],
            "job_training_accuracy": [],
            "job_training_progress": [],
            "job_schedule_moment": [],
            "job_generation_moment": [],
            "job_elapsed_time": [],
            "job_required_epoch": [],
            "job_passed_epoch": [],
            "job_model_complexity": [],
            "job_dataset_complexity": [],
            "job_batch_size": [],
            "job_learning_rate": [],
        }
        for port in ports:
            port_metrics_for_this_port = get_port_metrics(port)
            for metric, values in port_metrics_for_this_port.items():
                port_metrics[metric].extend(values)
        metrics_data = {**node_metrics, **port_metrics}
        time.sleep(1)


def flatten_metrics(metrics):
    return np.concatenate([np.array(v, dtype=np.float32) for v in metrics.values()])


@app.route("/state", methods=["GET"])
def serve_metrics():
    flattened_state = flatten_metrics(metrics_data)
    return jsonify(flattened_state.tolist())


def run_server():
    app.run(host="0.0.0.0", port=9807, debug=False, use_reloader=False)


def signal_handler(sig, frame):
    print("Shutting down...")
    exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    updater_thread = threading.Thread(target=update_metrics, daemon=True)
    updater_thread.start()
    run_server()

