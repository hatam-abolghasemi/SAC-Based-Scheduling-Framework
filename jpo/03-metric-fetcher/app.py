import requests
import time
from flask import Flask, Response

PROMETHEUS_URL = 'http://172.16.104.253:9090'
NODE_CPU_UTILIZATION_QUERY = 'avg(100 - (avg by(instance) (irate(node_cpu_seconds_total{job="node-exporter", mode="idle"}[5m])) * 100))'
NODE_MEMORY_UTILIZATION_QUERY = 'avg((1 - (node_memory_MemAvailable_bytes{job="node-exporter"} / node_memory_MemTotal_bytes{job="node-exporter"})) * 100)'
CONTAINER_CPU_UTILIZATION_QUERY = 'sum(rate(container_cpu_usage_seconds_total{job="cadvisor",image!=""}[1m])) by (name, image, container_label_dataset, container_label_framework, container_label_model, container_label_batch_size, container_label_learning_rate, container_label_num_epochs)'

app = Flask(__name__)

metrics_data = ''

def fetch_node_cpu_utilization():
    global metrics_data
    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': NODE_CPU_UTILIZATION_QUERY})
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            results = data['data']['result']
            if results:
                value = float(results[0]['value'][1])
                unix_time = int(time.time())
                metrics_data += f'{unix_time} node_cpu_utilization {value:.2f}%\n'

def fetch_node_memory_utilization():
    global metrics_data
    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': NODE_MEMORY_UTILIZATION_QUERY})
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            results = data['data']['result']
            if results:
                value = float(results[0]['value'][1])
                unix_time = int(time.time())
                metrics_data += f'{unix_time} node_memory_utilization {value:.2f}%\n'

def fetch_container_cpu_utilization():
    global metrics_data
    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': CONTAINER_CPU_UTILIZATION_QUERY})
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            results = data['data']['result']
            unix_time = int(time.time())
            for result in results:
                # Extract labels, defaulting to 'unknown_<label_name>' if not present
                labels = {
                    'name': result['metric'].get('name', 'unknown_container'),
                    'image': result['metric'].get('image', 'unknown_image'),
                    'dataset': result['metric'].get('container_label_dataset', 'unknown_dataset'),
                    'framework': result['metric'].get('container_label_framework', 'unknown_framework'),
                    'model': result['metric'].get('container_label_model', 'unknown_model'),
                    'batch_size': result['metric'].get('container_label_batch_size', 'unknown_batch_size'),
                    'learning_rate': result['metric'].get('container_label_learning_rate', 'unknown_learning_rate'),
                    'num_epochs': result['metric'].get('container_label_num_epochs', 'unknown_num_epochs'),
                }

                value = float(result['value'][1])
                metrics_data += (f'{unix_time} container_cpu_utilization('
                                 f'image:{labels["image"]},container:{labels["name"]},'
                                 f'dataset:{labels["dataset"]},framework:{labels["framework"]},'
                                 f'model:{labels["model"]},batch_size:{labels["batch_size"]},'
                                 f'learning_rate:{labels["learning_rate"]},num_epochs:{labels["num_epochs"]}) '
                                 f'{value:.2f}%\n')

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return Response(metrics_data, content_type='text/plain')

if __name__ == "__main__":
    # Start a background thread to fetch metrics periodically
    def fetch_metrics():
        while True:
            global metrics_data
            metrics_data = ''  # Reset metrics data before each fetch cycle
            fetch_node_cpu_utilization()
            fetch_node_memory_utilization()
            fetch_container_cpu_utilization()
            time.sleep(15)

    import threading
    threading.Thread(target=fetch_metrics, daemon=True).start()

    # Start the Flask application
    app.run(host='0.0.0.0', port=4223)


