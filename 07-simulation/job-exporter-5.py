import os
import time
import logging
from prometheus_client import start_http_server, Gauge, Counter, REGISTRY
import argparse
import threading
import requests
import re
import sqlite3


# Step 1: Initialization --------------------------------------------------------------------------------------------------
# Step 1.1: Define Prometheus metrics globally, outside any functions so they are only initialized once
container_cpu_usage = Gauge('container_used_cpu', 'CPU usage for the container', ['generation_id', 'job_id', 'node'])
container_gpu_usage = Gauge('container_used_gpu', 'GPU usage for the container', ['generation_id', 'job_id', 'node'])
container_mem_usage = Gauge('container_used_mem', 'Memory usage for the container', ['generation_id', 'job_id', 'node'])
job_training_loss = Gauge('job_training_loss', 'Cross-entropy loss for the training job', ['generation_id', 'job_id', 'node'])
job_training_accuracy = Gauge('job_training_accuracy', 'Accuracy for the training job', ['generation_id', 'job_id', 'node'])
job_training_progress = Gauge('job_training_progress', 'Progress for the training job', ['generation_id', 'job_id', 'node'])
job_schedule_moment = Gauge('job_schedule_moment', 'The moment the job was scheduled', ['generation_id', 'job_id', 'node'])
job_generation_moment = Gauge('job_generation_moment', 'The moment the job was generated', ['generation_id', 'job_id', 'node'])
job_elapsed_time = Gauge('job_elapsed_time', 'Elapsed time since the schedule moment', ['generation_id', 'job_id', 'node'])
job_required_epoch = Gauge('job_required_epoch', 'The number of required epochs for the job', ['generation_id', 'job_id', 'node'])
job_passed_epoch = Gauge('job_passed_epoch', 'The number of passed epochs for the job', ['generation_id', 'job_id', 'node'])
job_model_complexity_gauge = Gauge('job_model_complexity', 'Complexity level of the used model', ['generation_id', 'job_id', 'node'])
job_dataset_complexity_gauge = Gauge('job_dataset_complexity', 'Complexity level of the used dataset', ['generation_id', 'job_id', 'node'])
job_batch_size_gauge = Gauge('job_batch_size', 'The batch size for the job', ['generation_id', 'job_id', 'node'])
job_learning_rate_gauge = Gauge('job_learning_rate', 'The learning rate for the job', ['generation_id', 'job_id', 'node'])

# Step 1.2: Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Step 1.3: Fetch static data
def fetch_static_data(job_id, generation_id, node):
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT job_id, job_batch_size, job_learning_rate, job_dataset_complexity, job_model_complexity FROM jobs WHERE job_id=?', (job_id,))
    job = cursor.fetchone()
    if job:
        job_id, job_batch_size, job_learning_rate, job_dataset_complexity, job_model_complexity = job
        job_batch_size = int(job_batch_size)
        job_dataset_complexity = int(job_dataset_complexity)
        job_model_complexity = int(job_model_complexity)
        job_learning_rate = round(float(job_learning_rate), 2)
        job_model_complexity_gauge.labels(generation_id=generation_id, job_id=job_id, node=node).set(job_model_complexity)
        job_dataset_complexity_gauge.labels(generation_id=generation_id, job_id=job_id, node=node).set(job_dataset_complexity)
        job_batch_size_gauge.labels(generation_id=generation_id, job_id=job_id, node=node).set(job_batch_size)
        job_learning_rate_gauge.labels(generation_id=generation_id, job_id=job_id, node=node).set(job_learning_rate)
        print(f"job_id: {job_id}, job_batch_size: {job_batch_size}, job_learning_rate: {job_learning_rate}, "
              f"job_dataset_complexity: {job_dataset_complexity}, job_model_complexity: {job_model_complexity}")
    else:
        print(f"No job found with job_id: {job_id}")
    conn.close()


# Step 1.4: Increment the job_schedule_moment and job_generation_moment every second
def init(generation_id, job_id, node, schedule_moment_value, generation_moment_value, required_epoch_value):
    job_schedule_moment.labels(generation_id=generation_id, job_id=job_id, node=node).set(schedule_moment_value)
    job_generation_moment.labels(generation_id=generation_id, job_id=job_id, node=node).set(generation_moment_value)
    job_required_epoch.labels(generation_id=generation_id, job_id=job_id, node=node).set(required_epoch_value)
    job_training_loss.labels(generation_id=generation_id, job_id=job_id, node=node).set(0)
    job_training_accuracy.labels(generation_id=generation_id, job_id=job_id, node=node).set(0)
    job_training_progress.labels(generation_id=generation_id, job_id=job_id, node=node).set(0)
    job_passed_epoch.labels(generation_id=generation_id, job_id=job_id, node=node).set(0)


# Step 2: Resource usage simulation functions --------------------------------------------------------------------------------------------------
# Step 2.1: Creating dataset
def get_stage_requirements(passed_epochs):
    if passed_epochs < 10:
        mem = [2.0, 2.2, 2.5]
        cpu = [1.0, 0.3, 0.2]
        gpu = [80, 600, 50]
    elif passed_epochs < 50:
        mem = [2.2, 2.4, 2.7]
        cpu = [1.1, 0.4, 0.3]
        gpu = [90, 650, 70]
    elif passed_epochs < 100:
        mem = [2.4, 2.6, 2.9]
        cpu = [1.2, 0.5, 0.3]
        gpu = [100, 700, 90]
    elif passed_epochs < 300:
        mem = [2.6, 2.8, 3.1]
        cpu = [1.3, 0.6, 0.4]
        gpu = [110, 750, 110]
    else:
        mem = [2.8, 3.0, 3.3]
        cpu = [1.4, 0.7, 0.4]
        gpu = [120, 800, 130]

    return [{'cpu': cpu[i], 'memory': mem[i], 'gpu': gpu[i]} for i in range(3)]


def get_required_resource(stage):
    return {
        'cpu': stage['cpu'],
        'gpu': stage['gpu'],
        'memory': stage['memory']
    }


def get_available_resource(worker_node):
    url = "http://0.0.0.0:9904/metrics"
    try:
        response = requests.get(url)
        response.raise_for_status()
        metrics = response.text.split("\n")
        free_cpu, free_gpu, free_mem = None, None, None
        for line in metrics:
            if f'node_free_cpu{{name="{worker_node}"}}' in line:
                free_cpu = float(re.findall(r'[-+]?[0-9]*\.?[0-9]+$', line)[-1])
            elif f'node_free_gpu{{name="{worker_node}"}}' in line:
                free_gpu = float(re.findall(r'[-+]?[0-9]*\.?[0-9]+$', line)[-1])
            elif f'node_free_mem{{name="{worker_node}"}}' in line:
                free_mem = float(re.findall(r'[-+]?[0-9]*\.?[0-9]+$', line)[-1])
        if None in (free_cpu, free_gpu, free_mem):
            logging.warning(f"Some metrics for {worker_node} could not be found.")
        else:
            logging.info(f"Available resources for {worker_node}: CPU={free_cpu}, GPU={free_gpu}, Memory={free_mem}")
        return {'cpu': free_cpu, 'gpu': free_gpu, 'memory': free_mem}
    except requests.RequestException as e:
        logging.error(f"Error fetching available resource: {e}")
        return None

def get_allowed_resource(worker_node, stage_index, stage):
    available_resource = get_available_resource(worker_node)
    if not available_resource:
        logging.error("Failed to fetch available resource.")
        return None, stage
    required_resource = get_required_resource(stage)
    allowed_resource = {'cpu': 0, 'gpu': 0, 'memory': 0}
    insufficient = any(available_resource[res] < required_resource[res] for res in ['cpu', 'gpu', 'memory'])
    if insufficient:
        for res in allowed_resource:
            allowed_resource[res] = 0
        logging.warning(f"Insufficient resources: Required={required_resource}, Available={available_resource}. Repeating stage {stage_index}")
        return allowed_resource, max(0, stage_index - 1)
    allowed_resource = required_resource.copy()
    return allowed_resource, stage_index

# Step 2.2: Send calculated resource usage metrics to Prometheus
def expose_resource_usage_metrics(generation_id, job_id, node, allowed_resource):
    logger.info(f"Used resource for job {job_id} on node {node} with generation ID {generation_id}: {allowed_resource}")
    container_cpu_usage.labels(generation_id, job_id, node).set(allowed_resource['cpu'])
    container_gpu_usage.labels(generation_id, job_id, node).set(allowed_resource['gpu'])
    container_mem_usage.labels(generation_id, job_id, node).set(allowed_resource['memory'])


# Step 2.3: Simulate resource usage for each stage and proceed only when all resources are consumed
def simulate_resource_usage_for_stages(generation_id, job_id, node, passed_epochs):
    stages = get_stage_requirements(passed_epochs)
    stage_index = 0
    while stage_index < len(stages):
        stage = stages[stage_index]
        required_resource = get_required_resource(stage)
        logger.info(f"Simulating resource usage for stage {stage_index + 1}, job {job_id} on node {node} with generation ID {generation_id}: {required_resource}")
        required_resource = get_required_resource(stage)
        allowed_resource, updated_stage = get_allowed_resource(node, stage_index, stage)
        expose_resource_usage_metrics(generation_id, job_id, node, allowed_resource)
        if updated_stage < stage_index:
            logger.warning(f"Stage {stage_index + 1} failed. Repeating stage {stage_index + 1}.")
        else:
            stage_index = updated_stage + 1  
        time.sleep(1)
        logger.info(f"Stage {stage_index} of epoch {passed_epochs} completed.")

# Step 3: Training job simulation functions --------------------------------------------------------------------------------------------------
# Step 3.1: Calculate training loss based on passed_epoch and progress_percentage
def get_training_loss(passed_epochs, progress_percentage):
    loss_matrix = {
        10: [2.40, 2.40, 2.40, 2.40, 2.40],
        50: [2.00, 1.95, 1.90, 1.85, 1.90],
        100: [1.70, 1.65, 1.60, 1.55, 1.60],
        300: [1.40, 1.35, 1.30, 1.25, 1.30],
        500: [1.20, 1.15, 1.10, 1.05, 1.10],
    }
    for threshold in sorted(loss_matrix.keys(), reverse=True):
        if passed_epochs >= threshold:
            progress_index = min(int(progress_percentage // 20), 4)
            return loss_matrix[threshold][progress_index]
    return 2.40


def get_training_accuracy(passed_epochs, progress_percentage):
    accuracy_matrix = {
        10: [55.0, 55.5, 56.0, 56.5, 56.0],
        50: [66.0, 67.0, 68.0, 69.0, 68.5],
        100: [72.0, 73.0, 74.0, 75.0, 75.5],
        300: [77.0, 78.0, 79.0, 80.0, 81.0],
        500: [80.0, 81.0, 82.0, 83.0, 84.0],
    }
    for threshold in sorted(accuracy_matrix.keys(), reverse=True):
        if passed_epochs >= threshold:
            progress_index = min(int(progress_percentage // 20), 4)
            return accuracy_matrix[threshold][progress_index]
    return 55.0

# Step 3.3: Send calculated training metrics to Prometheus
def send_training_metrics(generation_id, job_id, node, training_loss, training_accuracy, progress_percentage):
    logger.info(f"Updating training metrics for job {job_id} on node {node} with generation ID {generation_id}: Loss={training_loss}, Accuracy={training_accuracy}")
    job_training_loss.labels(generation_id, job_id, node).set(training_loss)
    job_training_accuracy.labels(generation_id, job_id, node).set(training_accuracy)
    job_training_progress.labels(generation_id, job_id, node).set(progress_percentage)


# Step 4: Aggregate and expose all metrics for Prometheus --------------------------------------------------------------------------------------------------
def start_elapsed_time_counter(generation_id, job_id, node, schedule_moment_value):
    # job_elapsed_time.labels(generation_id=generation_id, job_id=job_id, node=node).set(schedule_moment_value)
    job_elapsed_time.labels(generation_id=generation_id, job_id=job_id, node=node).set(0)
    while True:
        time.sleep(1)
        job_elapsed_time.labels(generation_id=generation_id, job_id=job_id, node=node).inc()


def start_job_exporter(port):
    try:
        start_http_server(port)
        logger.info(f"Started job-exporter on port {port}")
    except OSError:
        logger.error(f"Port {port} is already in use. Choose another.")

# Step 5: Simulate the epoch process, handling both resource usage and training job metrics --------------------------------------------------------------------------------------------------
def simulate_epoch(passed_epochs, required_epochs, job_id, node, generation_id, schedule_moment, generation_moment):
    progress_percentage = int((passed_epochs / required_epochs) * 100)
    simulate_resource_usage_for_stages(generation_id, job_id, node, passed_epochs)
    training_loss = round(float(get_training_loss(passed_epochs, progress_percentage)), 2)
    training_accuracy = get_training_accuracy(passed_epochs, progress_percentage)
    send_training_metrics(generation_id, job_id, node, training_loss, training_accuracy, progress_percentage)
    job_passed_epoch.labels(generation_id=generation_id, job_id=job_id, node=node).set(passed_epochs)
    job_required_epoch.labels(generation_id=generation_id, job_id=job_id, node=node).set(required_epochs)
            
def deregister_unwanted_metrics():
    unwanted_metrics = [
        'python_gc_objects_uncollectable_total',
        'python_gc_collections_total',
        'python_info',
        'process_virtual_memory_bytes',
        'process_resident_memory_bytes',
        'process_start_time_seconds',
        'process_cpu_seconds_total',
        'process_open_fds',
        'process_max_fds'
    ]
    for metric_name in unwanted_metrics:
        if metric_name in REGISTRY._names_to_collectors:
            REGISTRY.unregister(REGISTRY._names_to_collectors[metric_name])

def main():
    deregister_unwanted_metrics()
    parser = argparse.ArgumentParser(description="Container Exporter")
    parser.add_argument('--port', type=int, required=True, help="Port for the container exporter")
    parser.add_argument('--generation_id', type=int, required=True, help="Generation ID")
    parser.add_argument('--node', type=str, required=True, help="Node name")
    parser.add_argument('--job_id', type=int, required=True, help="Job ID")
    parser.add_argument('--generation_moment', type=int, required=True, help="Generation Moment")
    parser.add_argument('--schedule_moment', type=int, required=True, help="Schedule Moment")
    parser.add_argument('--required_epochs', type=int, required=True, help="Required epochs")
    args = parser.parse_args()
    start_job_exporter(args.port)
    init(args.generation_id, args.job_id, args.node, args.schedule_moment, args.generation_moment, args.required_epochs)
    fetch_static_data(args.job_id, args.generation_id, args.node)
    elapsed_time_thread = threading.Thread(target=start_elapsed_time_counter, args=(args.generation_id, args.job_id, args.node, args.schedule_moment))
    elapsed_time_thread.daemon = True
    elapsed_time_thread.start()
    required_epochs = args.required_epochs
    passed_epochs = 1
    while passed_epochs <= required_epochs:
        simulate_epoch(
            passed_epochs, required_epochs, args.job_id, args.node, args.generation_id, args.schedule_moment, args.generation_moment
        )
        print(f"Epoch {passed_epochs}/{required_epochs} completed.")
        passed_epochs += 1
    print(f"All {required_epochs} epochs completed. Exiting successfully.")


if __name__ == '__main__':
    main()

