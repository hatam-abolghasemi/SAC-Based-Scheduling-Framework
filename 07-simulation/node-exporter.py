import requests
import time
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import sqlite3
import re

class Node:
    def __init__(self, node_type, name):
        self.name = name
        self.node_type = node_type
        self.total_cpu = 32 if node_type == 'worker' else 8
        self.total_memory = 96 if node_type == 'worker' else 16
        self.total_gpu = 1 if node_type == 'worker' else 0
        self.jobs = []  # List of jobs deployed on this node

    def update_jobs(self, jobs):
        self.jobs = jobs

    def simulate_metrics(self):
        job_count = len(self.jobs)

        # Sum up resource usage based on the number of jobs and simulated resource consumption
        total_cpu_usage = sum(job['dl_requested_cpu'] for job in self.jobs) if job_count > 0 else 0
        total_mem_usage = sum(job['dl_requested_mem'] for job in self.jobs) if job_count > 0 else 0
        total_gpu_usage = sum(job['dl_requested_gpu'] for job in self.jobs) if job_count > 0 else 0

        # Calculate CPU, Memory, and GPU usage as percentages
        cpu_usage = min(100, (total_cpu_usage / self.total_cpu) * 100)
        mem_usage = min(100, (total_mem_usage / self.total_memory) * 100)
        gpu_usage = min(100, (total_gpu_usage / self.total_gpu) * 100) if self.total_gpu else 0

        # Simulating CPU load (could be a random fluctuation)
        cpu_load1 = round(random.uniform(0.5, 2.0), 2)

        # Returning the metrics in Prometheus format
        metrics = [
            f'node_cpu_load1{{name="{self.name}",type="{self.node_type}"}} {cpu_load1}',
            f'node_container_count{{name="{self.name}",type="{self.node_type}"}} {job_count}',
            f'node_gpu_utilization{{name="{self.name}",type="{self.node_type}"}} {gpu_usage}',
            f'node_mem_utilization{{name="{self.name}",type="{self.node_type}"}} {mem_usage}',
            f'node_cpu_utilization{{name="{self.name}",type="{self.node_type}"}} {cpu_usage}'
        ]
        return '\n'.join(metrics)

def parse_memory(value):
    """Parse memory values with units (e.g., '16Gi', '4Mi') and return the integer in MiB."""
    match = re.match(r'(\d+)(Gi|Mi|Ki)', value)
    if match:
        size = int(match.group(1))
        unit = match.group(2)
        if unit == 'Gi':
            return size * 1024  # Convert GiB to MiB
        elif unit == 'Mi':
            return size  # Already in MiB
        elif unit == 'Ki':
            return size // 1024  # Convert KiB to MiB
    raise ValueError(f"Invalid memory format: {value}")

# Read the job information from the database (just once)
def load_jobs_from_db():
    jobs = []
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()

    cursor.execute('SELECT job_id, dl_batch_size, dl_learning_rate, dl_expected_time, dl_requested_cpu, dl_requested_mem, dl_requested_gpu, dl_dataset, dl_framework, dl_model FROM jobs')
    rows = cursor.fetchall()
    for row in rows:
        job = {
            'job_id': row[0],
            'dl_batch_size': row[1],
            'dl_learning_rate': row[2],
            'dl_expected_time': row[3],
            'dl_requested_cpu': int(row[4]),  # Ensure it's an integer
            'dl_requested_mem': parse_memory(row[5]),  # Parse memory with units
            'dl_requested_gpu': int(row[6]),  # Ensure it's an integer
            'dl_dataset': row[7],
            'dl_framework': row[8],
            'dl_model': row[9]
        }
        jobs.append(job)
    conn.close()
    return jobs

# Initialize node list
nodes = [
    Node('master', f'k8s-master-{i+1}') for i in range(5)
] + [
    Node('worker', f'k8s-worker-{i+1}') for i in range(13)
]
node_dict = {node.name: node for node in nodes}

# Preload the jobs into memory once
jobs_list = load_jobs_from_db()

def fetch_pods():
    try:
        response = requests.get('http://0.0.0.0:9901/pods')
        if response.status_code == 200:
            # Process each line in the response, assuming it's newline-delimited JSON
            lines = response.text.splitlines()
            for line in lines:
                try:
                    pod = json.loads(line)
                    node_name = pod['node']
                    if node_name in node_dict:
                        # Simulate the deployment of jobs on nodes
                        deployed_jobs = [job for job in jobs_list if job['job_id'] == pod['job_id']]
                        node_dict[node_name].update_jobs(deployed_jobs)  # Update with the jobs for that node
                except json.JSONDecodeError as e:
                    print(f"Error parsing pod line: {line} - {e}")
    except Exception as e:
        print(f"Error fetching pods: {e}")

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            fetch_pods()  # Fetch current pods and simulate their effects on nodes
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

