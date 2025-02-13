import requests
import time
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Node:
    def __init__(self, node_type, name):
        self.name = name
        self.node_type = node_type
        self.total_cpu = 32 if node_type == 'worker' else 8
        self.total_memory = 96 if node_type == 'worker' else 16
        self.total_gpu = 1 if node_type == 'worker' else 0
        self.pods = []

    def update_pods(self, pods):
        self.pods = pods

    def simulate_metrics(self):
        pod_count = len(self.pods)

        # Sum up resource usage based on the number of pods and simulated resource consumption
        total_cpu_usage = sum(pod['dl_requested_cpu'] for pod in self.pods) if pod_count > 0 else 0
        total_mem_usage = sum(pod['dl_requested_mem'] for pod in self.pods) if pod_count > 0 else 0
        total_gpu_usage = sum(pod['dl_requested_gpu'] for pod in self.pods) if pod_count > 0 else 0

        # Calculate CPU, Memory, and GPU usage as percentages
        cpu_usage = min(100, (total_cpu_usage / self.total_cpu) * 100)
        mem_usage = min(100, (total_mem_usage / self.total_memory) * 100)
        gpu_usage = min(100, (total_gpu_usage / self.total_gpu) * 100) if self.total_gpu else 0

        # Simulating CPU load (could be a random fluctuation)
        cpu_load1 = round(random.uniform(0.5, 2.0), 2)

        # Returning the metrics in Prometheus format
        metrics = [
            f'node_cpu_load1{{name="{self.name}",type="{self.node_type}"}} {cpu_load1}',
            f'node_container_count{{name="{self.name}",type="{self.node_type}"}} {pod_count}',
            f'node_gpu_utilization{{name="{self.name}",type="{self.node_type}"}} {gpu_usage}',
            f'node_mem_utilization{{name="{self.name}",type="{self.node_type}"}} {mem_usage}',
            f'node_cpu_utilization{{name="{self.name}",type="{self.node_type}"}} {cpu_usage}'
        ]
        return '\n'.join(metrics)

nodes = [
    Node('master', f'k8s-master-{i+1}') for i in range(5)
] + [
    Node('worker', f'k8s-worker-{i+1}') for i in range(13)
]
node_dict = {node.name: node for node in nodes}

def fetch_pods():
    try:
        response = requests.get('http://0.0.0.0:9901/pods')
        if response.status_code == 200:
            pods = json.loads(response.text)
            for pod in pods:
                node_name = pod['node']
                if node_name in node_dict:
                    # For simulation, assuming each pod's resource requirements
                    simulated_pod = {
                        'dl_requested_cpu': random.randint(1, 4),  # Random CPU requests between 1-4 CPUs
                        'dl_requested_mem': random.randint(4, 16),  # Random memory requests between 4-16 GB
                        'dl_requested_gpu': random.randint(0, 1),  # Random GPU requests, 0 or 1
                        'dl_batch_size': random.randint(16, 128),
                        'dl_learning_rate': random.uniform(0.001, 0.1),
                        'dl_expected_time': random.randint(30, 180),  # Simulated expected time in seconds
                        'dl_dataset': f'dataset_{random.randint(1, 10)}',
                        'dl_framework': random.choice(['TensorFlow', 'PyTorch']),
                        'dl_model': f'model_{random.randint(1, 5)}'
                    }
                    node_dict[node_name].update_pods([simulated_pod])  # Updating with simulated pod details
    except Exception as e:
        print(f"Error fetching pods: {e}")

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            fetch_pods()
            metrics = '\n'.join(node.simulate_metrics() for node in nodes)
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(metrics.encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 9904), MetricsHandler)
    print("Node exporter running on port 9904")
    server.serve_forever()

