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
    metadata = {}
    lines = metrics_text.strip().split('\n')
    for line in lines:
        if 'container_cpu_utilization' in line:
            metadata_line = line.split('(')[1].split(')')[0]
            meta_parts = metadata_line.split(',')
            for part in meta_parts:
                key_value = part.split(':', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    metadata[key.strip()] = value.strip()
    return metadata

def suggest_new_values(current_metrics, last_metrics):
    suggestions = {}
    for key, value in current_metrics.items():
        if 'node_cpu_utilization' in key or 'container_cpu_utilization' in key:
            if last_metrics and value > last_metrics.get(key, 0):
                suggestions['batch_size'] = suggestions.get('batch_size', 128) + 32
                suggestions['learning_rate'] = suggestions.get('learning_rate', 0.001) * 1.1
                suggestions['num_epochs'] = suggestions.get('num_epochs', 30) + 5
            else:
                suggestions['batch_size'] = max(32, suggestions.get('batch_size', 128) - 32)
                suggestions['learning_rate'] = suggestions.get('learning_rate', 0.001) * 0.9
                suggestions['num_epochs'] = max(1, suggestions.get('num_epochs', 30) - 5)
    return suggestions

def write_suggestions_to_config(suggestions, metadata, config_file='config.ini'):
    config = configparser.ConfigParser()

    # Create section name based on metadata
    section_name = f"{metadata.get('dataset', 'default')}_{metadata.get('model', 'default')}_{metadata.get('framework', 'default')}"
    if section_name not in config:
        config[section_name] = {}

    # Update parameters
    config[section_name].update({
        'batch_size': str(suggestions.get('batch_size', 128)),
        'learning_rate': str(suggestions.get('learning_rate', 0.001)),
        'num_epochs': str(suggestions.get('num_epochs', 30)),
        'dataset': metadata.get('dataset', 'unknown'),
        'model': metadata.get('model', 'unknown'),
        'framework': metadata.get('framework', 'unknown')
    })

    with open(config_file, 'w') as configfile:
        config.write(configfile)

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
        print(f"The current metrics text are: {current_metrics_text}")
        if current_metrics_text:
            current_metrics = parse_metrics(current_metrics_text)
            metadata = extract_metadata(current_metrics_text)
            print(f"Extracted metadata: {metadata}")

            # Generate suggestions based on metrics
            suggestions = suggest_new_values(current_metrics, last_metrics)

            # Update configuration file with new suggestions
            write_suggestions_to_config(suggestions, metadata)

            # Update last metrics
            last_metrics = current_metrics
        
        # Sleep before the next iteration
        time.sleep(15)

if __name__ == "__main__":
    main()

