import requests
import time
from flask import Flask, Response
import threading
import re

# Prometheus URL in the monitoring namespace
PROMETHEUS_URL = 'http://172.19.0.2:30090'

# Queries for node-level and container-level metrics
NODE_CPU_UTILIZATION_QUERY = '''
100 - (avg by (instance) (irate(node_cpu_seconds_total{job="node-exporter", mode="idle"}[5m])) * 100)
'''
NODE_MEMORY_UTILIZATION_QUERY = '''
(1 - (node_memory_MemAvailable_bytes{job="node-exporter"} / node_memory_MemTotal_bytes{job="node-exporter"})) * 100
'''
CONTAINER_CPU_UTILIZATION_QUERY = '''
sum(rate(container_cpu_usage_seconds_total{
    job="cadvisor", image!~"registry.k8s.io/pause:3.10|", namespace="dl"
}[1m])) by (container)
'''

app = Flask(__name__)
metrics_data = ''
lock = threading.Lock()  # Ensure thread-safe operations on metrics_data

# Regular expression to parse the app label
APP_LABEL_REGEX = r'^(?P<framework>[^_]+)_(?P<dataset>[^_]+)_(?P<model>[^_]+)_(?P<batch_size>\d+\.?\d*)_(?P<learning_rate>\d+\.\d+)_(?P<num_epochs>\d+)$'

def fetch_metrics_data(query, process_result_callback):
    """Fetch metrics data from Prometheus and process it."""
    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': query})
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            process_result_callback(data['data']['result'])

def process_node_cpu_utilization(results):
    """Process node CPU utilization results."""
    global metrics_data
    unix_time = int(time.time())
    with lock:
        for result in results:
            instance = result['metric'].get('instance', 'unknown_instance')
            value = float(result['value'][1])
            metrics_data += f'{unix_time} node_cpu_utilization(instance:{instance}) {value:.2f}%\n'

def process_node_memory_utilization(results):
    """Process node memory utilization results."""
    global metrics_data
    unix_time = int(time.time())
    with lock:
        for result in results:
            instance = result['metric'].get('instance', 'unknown_instance')
            value = float(result['value'][1])
            metrics_data += f'{unix_time} node_memory_utilization(instance:{instance}) {value:.2f}%\n'

def process_container_cpu_utilization(results):
    """Process container CPU utilization results."""
    global metrics_data
    unix_time = int(time.time())
    with lock:
        for result in results:
            # Extract container name from the metric
            container_name = result['metric'].get('container', 'unknown_container')
            
            # Adjust regex to match the expected format of the container name
            match = re.match(r'^(?P<framework>[^-]+)-(?P<dataset>[^-]+)-(?P<model>[^-]+)-(?P<batch_size>\d+)-(?P<learning_rate>\d+)-(?P<num_epochs>\d+)$', container_name)
            if match:
                labels = match.groupdict()
            else:
                labels = {
                    'framework': 'unknown_framework',
                    'dataset': 'unknown_dataset',
                    'model': 'unknown_model',
                    'batch_size': 'unknown_batch_size',
                    'learning_rate': 'unknown_learning_rate',
                    'num_epochs': 'unknown_num_epochs',
                }

            value = float(result['value'][1])
            metrics_data += (
                f'{unix_time} container_cpu_utilization(' 
                f'container:{container_name},framework:{labels["framework"]},dataset:{labels["dataset"]},'
                f'model:{labels["model"]},batch_size:{labels["batch_size"]},'
                f'learning_rate:0.{labels["learning_rate"]},num_epochs:{labels["num_epochs"]}) '
                f'{value:.2f}%\n'
            )

def fetch_metrics():
    """Periodic task to fetch and update metrics data."""
    while True:
        global metrics_data
        with lock:
            metrics_data = ''  # Reset metrics data before each fetch cycle
        
        fetch_metrics_data(NODE_CPU_UTILIZATION_QUERY, process_node_cpu_utilization)
        fetch_metrics_data(NODE_MEMORY_UTILIZATION_QUERY, process_node_memory_utilization)
        fetch_metrics_data(CONTAINER_CPU_UTILIZATION_QUERY, process_container_cpu_utilization)

        time.sleep(3)

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Expose metrics data through an HTTP endpoint."""
    with lock:
        return Response(metrics_data, content_type='text/plain')

if __name__ == "__main__":
    # Start a background thread to fetch metrics periodically
    threading.Thread(target=fetch_metrics, daemon=True).start()

    # Start the Flask application
    app.run(host='0.0.0.0', port=4223)


