import requests
import time
import configparser

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
                suggestions['batch_size'] = suggestions.get('batch_size', 128) - 32 if suggestions.get('batch_size', 128) > 32 else 32
                suggestions['learning_rate'] = suggestions.get('learning_rate', 0.001) * 0.9
                suggestions['num_epochs'] = suggestions.get('num_epochs', 30) - 5 if suggestions.get('num_epochs', 30) > 5 else 1
    return suggestions

def write_suggestions_to_config(suggestions, metadata, config_file='config.ini'):
    config = configparser.ConfigParser()
    section_name = f"{metadata['dataset']}_{metadata['model']}_{metadata['framework']}"
    if section_name not in config:
        config[section_name] = {}
    
    config[section_name] = {
        'batch_size': str(suggestions.get('batch_size', 128)),
        'learning_rate': str(suggestions.get('learning_rate', 0.001)),
        'num_epochs': str(suggestions.get('num_epochs', 30)),
        'dataset': metadata['dataset'],
        'model': metadata['model'],
        'framework': metadata['framework']
    }

    with open(config_file, 'w') as configfile:
        config.write(configfile)

def main():
    url = "http://0.0.0.0:4223/metrics"
    last_metrics = {}

    while True:
        current_metrics_text = fetch_metrics(url)
        if current_metrics_text:
            current_metrics = parse_metrics(current_metrics_text)
            metadata = extract_metadata(current_metrics_text)
            suggestions = suggest_new_values(current_metrics, last_metrics)
            write_suggestions_to_config(suggestions, metadata)
            last_metrics = current_metrics
        time.sleep(15)

if __name__ == "__main__":
    main()

