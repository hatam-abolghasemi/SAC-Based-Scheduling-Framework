import os
import time
import logging
from prometheus_client import start_http_server, Gauge, Counter
import argparse
import threading

# Step 1: Initialization --------------------------------------------------------------------------------------------------
# Step 1.1: Define Prometheus metrics globally, outside any functions so they are only initialized once
container_cpu_usage = Gauge('container_used_cpu', 'CPU usage for the container', ['generation_id', 'job_id', 'node'])
container_gpu_usage = Gauge('container_used_gpu', 'GPU usage for the container', ['generation_id', 'job_id', 'node'])
container_mem_usage = Gauge('container_used_mem', 'Memory usage for the container', ['generation_id', 'job_id', 'node'])
job_training_loss = Gauge('job_training_loss', 'Cross-entropy loss for the training job', ['generation_id', 'job_id', 'node'])
job_training_accuracy = Gauge('job_training_accuracy', 'Accuracy for the training job', ['generation_id', 'job_id', 'node'])
job_schedule_moment = Gauge('job_schedule_moment', 'The moment the job was scheduled', ['generation_id', 'job_id', 'node'])
job_generation_moment = Gauge('job_generation_moment', 'The moment the job was generated', ['generation_id', 'job_id', 'node'])
job_elapsed_time = Gauge('job_elapsed_time', 'Elapsed time since the schedule moment', ['generation_id', 'job_id', 'node'])
job_required_epoch = Gauge('job_required_epoch', 'The number of required epochs for the job', ['generation_id', 'job_id', 'node'])  # Static value
job_passed_epoch = Gauge('job_passed_epoch', 'The number of passed epochs for the job', ['generation_id', 'job_id', 'node'])  # Dynamic value


# Step 1.2: Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Step 1.3: Fetch job information and initialize values --------------------------------------------------------------------------------------------------
def initialize_job(job_id, node, required_epochs, generation_moment, schedule_moment):
    # Define configurable variables as environment variables
    ENVIRONMENT_VARIABLES = {
        'JOB_ID': job_id,
        'NODE': node,
        'REQUIRED_EPOCHS': required_epochs,
        'GENERATION_MOMENT': generation_moment,
        'SCHEDULE_MOMENT': schedule_moment
    }

    # Print initialization details
    logger.info(f"Initializing job {ENVIRONMENT_VARIABLES['JOB_ID']} on node {ENVIRONMENT_VARIABLES['NODE']} for {ENVIRONMENT_VARIABLES['REQUIRED_EPOCHS']} epochs.")
    
    # Generate a generation ID based on timestamp
    generation_id = f"{int(time.time()) % 1000:03}"
    job_required_epoch.labels(generation_id=generation_id, job_id=job_id, node=node).set(required_epochs)
    
    return ENVIRONMENT_VARIABLES, generation_id


# Step 1.4: Increment the job_schedule_moment and job_generation_moment every second
def expose_moments(generation_id, job_id, node, schedule_moment_value, generation_moment_value):
    # Expose the schedule_moment and generation_moment as static values
    job_schedule_moment.labels(generation_id=generation_id, job_id=job_id, node=node).set(schedule_moment_value)
    job_generation_moment.labels(generation_id=generation_id, job_id=job_id, node=node).set(generation_moment_value)


# Step 2: Resource usage simulation functions --------------------------------------------------------------------------------------------------
def get_stage_requirements(passed_epochs):
    # Define thresholds based on the passed epochs for dynamic resource allocation
    if passed_epochs < 10:
        mem = [4.5, 5.0, 5.5, 1.0]
        cpu = [2.5, 0.5, 0.7, 0.3]
        gpu = [200, 1800, 2200, 100]
    elif passed_epochs < 50:
        mem = [5.0, 5.1, 5.6, 1.2]
        cpu = [2.7, 0.5, 0.8, 0.4]
        gpu = [220, 1820, 2230, 120]
    elif passed_epochs < 100:
        mem = [5.2, 5.2, 5.8, 1.3]
        cpu = [2.8, 0.5, 0.9, 0.4]
        gpu = [230, 1850, 2250, 150]
    elif passed_epochs < 300:
        mem = [5.5, 5.4, 6.0, 1.5]
        cpu = [2.9, 0.6, 1.0, 0.5]
        gpu = [250, 1880, 2280, 170]
    else:
        mem = [5.7, 5.5, 6.2, 1.6]
        cpu = [3.0, 0.7, 1.1, 0.5]
        gpu = [270, 1900, 2300, 180]

    # Calculate dynamic stage resource requirements based on passed epochs
    return [{'cpu': cpu[i], 'memory': mem[i], 'gpu': gpu[i]} for i in range(4)]


def calculate_resource_usage(stage):
    return {
        'cpu': stage['cpu'],
        'gpu': stage['gpu'],
        'memory': stage['memory']
    }


# Step 2.3: Send calculated resource usage metrics to Prometheus
def send_resource_metrics(generation_id, job_id, node, resource_metrics):
    logger.info(f"Sending resource metrics for job {job_id}, node {node}, generation {generation_id}: {resource_metrics}")
    container_cpu_usage.labels(generation_id, job_id, node).set(resource_metrics['cpu'])
    container_gpu_usage.labels(generation_id, job_id, node).set(resource_metrics['gpu'])
    container_mem_usage.labels(generation_id, job_id, node).set(resource_metrics['memory'])


# Step 2.4: Simulate resource usage for each stage and proceed only when all resources are consumed
def simulate_resource_usage_for_stages(generation_id, job_id, node, passed_epochs):
    stages = get_stage_requirements(passed_epochs)
    for stage_index, stage in enumerate(stages):
        resource_metrics = calculate_resource_usage(stage)
        logger.info(f"Simulating resource usage for stage {stage_index + 1}, job {job_id}, node {node}, generation {generation_id}: {resource_metrics}")
        
        # Simulate resource consumption for this stage
        send_resource_metrics(generation_id, job_id, node, resource_metrics)
        
        # Sleep for one second to simulate the resource consumption time
        time.sleep(1)

        # Proceed to next stage if all resources are consumed for this stage
        logger.info(f"Stage {stage_index + 1} of epoch {passed_epochs} completed.")


# Step 3: Training job simulation functions --------------------------------------------------------------------------------------------------
# Step 3.1: Calculate training loss based on passed_epoch and progress_percentage
def get_training_loss(passed_epochs, progress_percentage):
    loss_matrix = {
        10: [1.85, 1.85, 1.85, 1.85, 1.85],
        50: [1.40, 1.35, 1.30, 1.20, 1.30],
        100: [1.10, 1.05, 1.00, 0.90, 1.05],
        300: [0.85, 0.80, 0.75, 0.70, 0.72],
        500: [0.75, 0.72, 0.70, 0.65, 0.72],
    }

    for threshold in sorted(loss_matrix.keys(), reverse=True):
        if passed_epochs >= threshold:
            progress_index = min(int(progress_percentage // 20), 4)
            return loss_matrix[threshold][progress_index]

    return 1.85


# Step 3.2: Calculate training accuracy based on passed_epoch and progress_percentage
def get_training_accuracy(passed_epochs, progress_percentage):
    accuracy_matrix = {
        10: [65.0, 65.5, 66.0, 66.5, 65.0],
        50: [75.0, 75.5, 77.0, 78.0, 75.5],
        100: [80.5, 81.0, 81.5, 82.0, 82.0],
        300: [85.5, 86.5, 87.0, 88.5, 89.5],
        500: [89.0, 89.5, 90.0, 91.0, 92.0],
    }

    for threshold in sorted(accuracy_matrix.keys(), reverse=True):
        if passed_epochs >= threshold:
            progress_index = min(int(progress_percentage // 20), 4)
            return accuracy_matrix[threshold][progress_index]

    return 65.0


# Step 3.3: Send calculated training metrics to Prometheus
def send_training_metrics(generation_id, job_id, node, training_loss, training_accuracy):
    logger.info(f"Sending training metrics for job {job_id}, node {node}, generation {generation_id}: Loss={training_loss}, Accuracy={training_accuracy}")
    job_training_loss.labels(generation_id, job_id, node).set(training_loss)
    job_training_accuracy.labels(generation_id, job_id, node).set(training_accuracy)


# Step 4: Aggregate and expose all metrics for Prometheus --------------------------------------------------------------------------------------------------
def start_elapsed_time_counter(generation_id, job_id, node, schedule_moment_value):
    # Set the initial value of elapsed_time to the value of schedule_moment
    #job_elapsed_time.labels(generation_id=generation_id, job_id=job_id, node=node).set(schedule_moment_value)
    job_elapsed_time.labels(generation_id=generation_id, job_id=job_id, node=node).set(0)

    # Increment the elapsed_time every second
    while True:
        time.sleep(1)
        job_elapsed_time.labels(generation_id=generation_id, job_id=job_id, node=node).inc()


def aggregate_metrics(generation_id, job_id, node, resource_metrics, training_loss, training_accuracy, schedule_moment, generation_moment, passed_epochs):
    if resource_metrics:
        send_resource_metrics(generation_id, job_id, node, resource_metrics)

    if training_loss is not None and training_accuracy is not None:
        send_training_metrics(generation_id, job_id, node, training_loss, training_accuracy)

    job_passed_epoch.labels(generation_id=generation_id, job_id=job_id, node=node).set(passed_epochs)


def start_prometheus_server(port):
    """Start the Prometheus server on the given port."""
    # Check if server is already started
    try:
        start_http_server(port)  # Use the port received from the flag
        logger.info(f"Started Prometheus server on port {port}")
    except OSError:
        logger.error(f"Port {port} is already in use. Choose another.")

# Step 5: Simulate the epoch process, handling both resource usage and training job metrics --------------------------------------------------------------------------------------------------
def simulate_epoch(passed_epochs, required_epochs, job_id, node, generation_id, schedule_moment, generation_moment):
    # Calculate progress percentage based on passed_epochs and required_epochs
    progress_percentage = (passed_epochs / required_epochs) * 100

    # Step 2: Resource usage simulation
    simulate_resource_usage_for_stages(generation_id, job_id, node, passed_epochs)

    # Step 3: Training job simulation (Send metrics after resource consumption)
    stages = get_stage_requirements(passed_epochs)
    for stage in stages:
        if all(stage.values()):
            # Only send resource metrics if they exist
            resource_metrics = calculate_resource_usage(stage)
            send_resource_metrics(generation_id, job_id, node, resource_metrics)

            # Send training metrics
            training_loss = get_training_loss(passed_epochs, progress_percentage)
            training_accuracy = get_training_accuracy(passed_epochs, progress_percentage)
            send_training_metrics(generation_id, job_id, node, training_loss, training_accuracy)

    # Update passed_epoch metric after each epoch
    job_passed_epoch.labels(generation_id=generation_id, job_id=job_id, node=node).set(passed_epochs)
            

def main():
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description="Container Exporter")
    parser.add_argument('--port', type=int, required=True, help="Port for the container exporter")
    parser.add_argument('--generation_id', type=int, required=True, help="Generation ID")
    parser.add_argument('--node', type=str, required=True, help="Node name")
    parser.add_argument('--job_id', type=int, required=True, help="Job ID")
    parser.add_argument('--generation_moment', type=int, required=True, help="Generation Moment")
    parser.add_argument('--schedule_moment', type=int, required=True, help="Schedule Moment")
    parser.add_argument('--required_epochs', type=int, required=True, help="Required epochs")

    args = parser.parse_args()

    # Initialize the job with the passed arguments
    ENVIRONMENT_VARIABLES, generation_id = initialize_job(
        args.job_id, args.node, args.required_epochs, args.generation_moment, args.schedule_moment
    )
    
    # Start the Prometheus server before the epoch simulation
    start_prometheus_server(args.port)

    # Expose schedule_moment and generation_moment as static values
    expose_moments(generation_id, args.job_id, args.node, args.schedule_moment, args.generation_moment)
    
    # Start the elapsed time counter in a separate thread
    elapsed_time_thread = threading.Thread(target=start_elapsed_time_counter, args=(generation_id, args.job_id, args.node, args.schedule_moment))
    elapsed_time_thread.daemon = True  # Daemonize the thread so it will exit when the main program exits
    elapsed_time_thread.start()


    # Get the required epochs from the parsed arguments
    required_epochs = args.required_epochs
    passed_epochs = 0


    # Simulate the required number of epochs
    while passed_epochs < required_epochs:
        simulate_epoch(
            passed_epochs, required_epochs, ENVIRONMENT_VARIABLES['JOB_ID'], ENVIRONMENT_VARIABLES['NODE'], generation_id,
            args.schedule_moment, args.generation_moment  # Pass schedule_moment and generation_moment
        )
        passed_epochs += 1
        print(f"Epoch {passed_epochs}/{required_epochs} completed.")
    
    print(f"All {required_epochs} epochs completed. Exiting successfully.")

if __name__ == '__main__':
    main()

