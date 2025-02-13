import requests
import time
import ast
from flask import Flask, jsonify
from collections import deque

app = Flask(__name__)

# Global variable to store scheduled jobs and job queue
scheduled_jobs = []
job_queue = deque()
max_jobs_per_second = 10  # Max jobs to process per second

# List of worker nodes (adjust this as per your cluster configuration)
worker_nodes = ['k8s-worker-1', 'k8s-worker-2', 'k8s-worker-3', 'k8s-worker-4', 'k8s-worker-5', 'k8s-worker-6', 'k8s-worker-7', 'k8s-worker-8']

# Function to get node names (filter only worker nodes)
def fetch_nodes():
    response = requests.get('http://0.0.0.0:9901/nodes')
    if response.status_code == 200:
        nodes = response.text.splitlines()
        # Filter out only worker nodes
        worker_nodes_in_cluster = [node for node in nodes if node in worker_nodes]
        return worker_nodes_in_cluster
    else:
        print("Error retrieving nodes. Using predefined worker nodes.")
        return worker_nodes  # Default to predefined worker nodes if the fetch fails

# Fetch node metrics from node-exporter
def fetch_node_metrics():
    node_metrics = {}
    response = requests.get('http://localhost:9904/metrics')
    if response.status_code == 200:
        metrics_text = response.text
        nodes = set()  # To track the unique nodes from the metrics
        for line in metrics_text.splitlines():
            if 'node_cpu_utilization' in line or 'node_mem_utilization' in line or 'node_gpu_utilization' in line:
                parts = line.split(' ')
                metric_name, value = parts[0], float(parts[1])
                node_name = metric_name.split('{')[1].split(',')[0].split('=')[1].strip('"')
                if node_name not in node_metrics:
                    node_metrics[node_name] = {}
                metric_key = metric_name.split('{')[0]
                node_metrics[node_name][metric_key] = value
                nodes.add(node_name)
        # Only include worker nodes in the metrics
        return {node: metrics for node, metrics in node_metrics.items() if node in worker_nodes}
    else:
        print("Error retrieving node metrics.")
        return {}

def score_node(node_metrics):
    node_scores = {}
    for node, metrics in node_metrics.items():
        cpu_util = metrics.get('node_cpu_utilization', 0)
        mem_util = metrics.get('node_mem_utilization', 0)
        gpu_util = metrics.get('node_gpu_utilization', 0)
        score = cpu_util + mem_util + gpu_util
        node_scores[node] = score
    return node_scores

def metric_based_scheduler(jobs, node_metrics):
    node_scores = score_node(node_metrics)
    scheduled_jobs = []
    for job in jobs:
        if not node_scores:
            print("No available nodes to schedule the job.")
            break
        sorted_nodes = sorted(node_scores, key=node_scores.get)
        best_node = sorted_nodes[0]
        scheduled_jobs.append({'job_id': job['job_id'], 'node': best_node})

        # Post to deploy the job
        deploy_job(best_node, job['job_id'])

        node_scores[best_node] += 1
    return scheduled_jobs

def deploy_job(node, job_id):
    url = 'http://0.0.0.0:9901/deploy_job'
    payload = {
        'job_id': job_id,
        'node': node
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print(f"Successfully deployed job {job_id} to {node}.")
        else:
            print(f"Failed to deploy job {job_id} to {node}: {response.text}")
    except Exception as e:
        print(f"Error sending job {job_id} to {node}: {e}")

def fetch_jobs():
    response = requests.get('http://localhost:9902/jobs')
    if response.status_code == 200:
        job_lines = response.text.splitlines()
        jobs = []
        for line in job_lines:
            try:
                job = ast.literal_eval(line)
                jobs.append(job)
            except Exception as e:
                print(f"Error parsing job line: {line}, {e}")
        return jobs
    else:
        print(f"Error fetching jobs, status code: {response.status_code}")
        return []

@app.route('/jobs', methods=['GET'])
def get_scheduled_jobs():
    return jsonify(scheduled_jobs)

def schedule_jobs():
    global scheduled_jobs, job_queue
    while True:
        jobs = fetch_jobs()
        if jobs:
            # Add the new jobs to the queue
            for job in jobs:
                if job['job_id'] not in [j['job_id'] for j in job_queue]:
                    job_queue.append(job)

        # Process jobs in the queue with rate limiting
        jobs_to_schedule = []
        while job_queue and len(jobs_to_schedule) < max_jobs_per_second:
            jobs_to_schedule.append(job_queue.popleft())

        if jobs_to_schedule:
            node_metrics = fetch_node_metrics()
            if node_metrics:
                scheduled_jobs = metric_based_scheduler(jobs_to_schedule, node_metrics)
                time.sleep(1)  # Wait for 1 second before processing again

if __name__ == '__main__':
    from threading import Thread

    # Start the scheduler in a separate thread
    scheduler_thread = Thread(target=schedule_jobs)
    scheduler_thread.daemon = True
    scheduler_thread.start()

    # Run the Flask server to expose the scheduled jobs
    app.run(host='0.0.0.0', port=9903)

