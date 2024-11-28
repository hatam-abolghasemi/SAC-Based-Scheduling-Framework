import requests
import time
import configparser
from gym import Env

# URL for the metrics fetcher
url = "http://0.0.0.0:4223/metrics"

class K8sEnv(Env):
    def __init__(self):
        # Define action space (e.g., actions to optimize parameters)
        self.action_space = None
        # Initial state
        self.state = None
        # Fetch initial metrics
        self.current_metrics_text = fetch_metrics(url)

    def step(self, action):
        # Apply suggested actions to optimize parameters
        write_suggestions_to_config(action.suggestions, action.metadata)

        # Update state with new metrics
        self.state = extract_metadata(self.current_metrics_text)
        
        # Define reward and done logic if needed
        reward = calculate_reward(self.state)
        done = check_done_condition(self.state)
        
        return self.state, reward, done

def fetch_metrics(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metrics: {e}")
        return None

def parse_metrics(metrics_text):
    state = {}
    lines = metrics_text.strip().split('\n')
    for line in lines:
        if line:
            parts = line.split(' ')
            metric_name = parts[1]
            metric_value = parts[2].rstrip('%')
            state[metric_name] = float(metric_value)
    return state

def extract_metadata(metrics_text):
    all_metadata = []
    lines = metrics_text.strip().split('\n')
    for line in lines:
        if 'container_cpu_utilization' in line:
            metadata_line = line.split('(')[1].split(')')[0]
            meta_parts = metadata_line.split(',')
            metadata = {}
            for part in meta_parts:
                key_value = part.split(':', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    metadata[key.strip()] = value.strip()
            all_metadata.append(metadata)
    return all_metadata

def suggest_new_values(job_specific_metrics, last_job_metrics):
    """
    Generate new hyperparameter suggestions based on job-specific metrics.
    """
    # Extract container CPU utilization if available
    # Extract the container CPU utilization for the current job
    job_cpu_utilization = next(
        (value for key, value in job_specific_metrics.items() if "container_cpu_utilization" in key),
        0  # Default to 0 if no matching key is found
        )

    print(f"Suggesting based on CPU utilization: {job_cpu_utilization}")

    # Adjust hyperparameters based on utilization
    new_batch_size = int(128 + 32 * (1 - job_cpu_utilization / 100))
    new_learning_rate = 0.001 * (1 + job_cpu_utilization / 1000)
    new_num_epochs = 30 + int(job_cpu_utilization / 20)

    # Ensure reasonable ranges
    new_batch_size = max(32, min(512, new_batch_size))
    new_learning_rate = max(0.0001, min(0.01, new_learning_rate))
    new_num_epochs = max(10, min(100, new_num_epochs))

    return {
        "batch_size": new_batch_size,
        "learning_rate": new_learning_rate,
        "num_epochs": new_num_epochs,
    }


def write_suggestions_to_config(suggestions, metadata, config_file='config.ini'):
    config = configparser.ConfigParser()
    print(f"the metadata in `write_` is: {metadata}")
    # Use dataset, model, and framework to create a unique section name
    section_name = f"{metadata.get('dataset', 'default')}_{metadata.get('model', 'default')}_{metadata.get('framework', 'default')}"
    
    # Read the existing config file (if it exists)
    try:
        config.read(config_file)
    except Exception as e:
        print(f"Error reading config file: {e}")

    # Ensure the section exists
    if section_name not in config:
        config[section_name] = {}
    
    # Update the section with new suggestions
    config[section_name].update({
        'batch_size': str(suggestions.get('batch_size', 128)),
        'learning_rate': str(suggestions.get('learning_rate', 0.001)),
        'num_epochs': str(suggestions.get('num_epochs', 30)),
        'dataset': metadata['dataset'],
        'model': metadata['model'],
        'framework': metadata['framework']
    })

    # Write changes back to the config file
    try:
        with open(config_file, 'w') as configfile:
            config.write(configfile)
    except Exception as e:
        print(f"Error writing to config file: {e}")

def calculate_reward(state):
    # Define logic for reward calculation
    # Example: lower CPU utilization results in a higher reward
    cpu_util = state.get('container_cpu_utilization', 100.0)
    return max(0, 100.0 - cpu_util)  # Higher reward for lower utilization

def check_done_condition(state):
    # Example: If CPU utilization is below a threshold, optimization is complete
    return state.get('container_cpu_utilization', 100.0) < 10.0

def main():
    last_metrics = {}

    while True:
        # Fetch current metrics
        current_metrics_text = fetch_metrics(url)
        if current_metrics_text:
            current_metrics = parse_metrics(current_metrics_text)
            all_metadata = extract_metadata(current_metrics_text)
            print(f"Extracted metadata: {all_metadata}")

            for metadata in all_metadata:
                job_key = metadata['container']  # Unique identifier for the job

                # Filter the current metrics to only include the current job's container
                job_specific_metrics = {
                    key: value
                    for key, value in current_metrics.items()
                    if job_key in key  # Match container name in metrics key
                }

                print(f"Processing job: {job_key}")
                print(f"Job-specific metrics: {job_specific_metrics}")

                # Generate suggestions based on job-specific metrics
                suggestions = suggest_new_values(job_specific_metrics, last_metrics.get(job_key, {}))

                # Update configuration file with suggestions for this job
                write_suggestions_to_config(suggestions, metadata)

                # Save current metrics for this job
                last_metrics[job_key] = job_specific_metrics


        # Sleep before the next iteration
        time.sleep(15)

if __name__ == "__main__":
    main()

