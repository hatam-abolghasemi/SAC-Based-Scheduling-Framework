import requests
import time
import threading
from flask import Flask, Response

PROMETHEUS_URL = 'http://172.16.104.253:9090'

app = Flask(__name__)
metrics_data = ''


def query_prometheus(query):
    """Helper function to send a query to Prometheus and return results."""
    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': query})
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            return data['data']['result']
    return []


def get_node_cpu_utilization():
    """Fetch node CPU utilization for each node."""
    query = 'avg by(instance) (100 - (irate(node_cpu_seconds_total{job="node-exporter", mode="idle"}[5m]) * 100))'
    results = query_prometheus(query)
    metrics = ''
    for result in results:
        node_name = result['metric'].get('instance', 'unknown_node')
        value = float(result['value'][1])
        metrics += f'node_cpu_utilization{{node="{node_name}"}} {value:.2f}%\n'
    return metrics


def get_node_memory_utilization():
    """Fetch node memory utilization per node."""
    query = 'avg by(instance) ((1 - (node_memory_MemAvailable_bytes{job="node-exporter"} / node_memory_MemTotal_bytes{job="node-exporter"})) * 100)'
    results = query_prometheus(query)
    metrics = ''
    for result in results:
        node_name = result['metric'].get('instance', 'unknown_node')
        value = float(result['value'][1])  # Fix: Process all results, not just the first one
        metrics += f'node_memory_utilization{{node="{node_name}"}} {value:.2f}%\n'
    return metrics


@app.route('/metrics', methods=['GET'])
def get_metrics():
    return Response(metrics_data, content_type='text/plain')


def fetch_metrics():
    """Periodically fetch metrics and update global state."""
    global metrics_data
    while True:
        temp_metrics = get_node_cpu_utilization()
        temp_metrics += get_node_memory_utilization()

        metrics_data = temp_metrics  # Atomically update metrics
        time.sleep(15)


if __name__ == "__main__":
    threading.Thread(target=fetch_metrics, daemon=True).start()
    app.run(host='0.0.0.0', port=4223)

