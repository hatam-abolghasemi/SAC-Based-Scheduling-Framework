import requests
import time
from flask import Flask, Response

PROMETHEUS_URL = 'http://127.0.0.1:9090'
CPU_UTILIZATION_QUERY = 'avg(100 - (avg by(instance) (irate(node_cpu_seconds_total{job="local_node_exporter", mode="idle"}[5m])) * 100))'
MEMORY_UTILIZATION_QUERY = 'avg((1 - (node_memory_MemAvailable_bytes{job="local_node_exporter"} / node_memory_MemTotal_bytes{job="local_node_exporter"})) * 100)'

app = Flask(__name__)

metrics_data = ''

def fetch_cpu_utilization():
    global metrics_data
    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': CPU_UTILIZATION_QUERY})
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            results = data['data']['result']
            if results:
                value = float(results[0]['value'][1])
                unix_time = int(time.time())
                metrics_data = f'{unix_time} cpu_utilization {value:.2f}%\n'

def fetch_memory_utilization():
    global metrics_data
    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': MEMORY_UTILIZATION_QUERY})
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            results = data['data']['result']
            if results:
                value = float(results[0]['value'][1])
                unix_time = int(time.time())
                metrics_data += f'{unix_time} memory_utilization {value:.2f}%\n'

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return Response(metrics_data, content_type='text/plain')

if __name__ == "__main__":
    # Start a background thread to fetch metrics periodically
    def fetch_metrics():
        while True:
            fetch_cpu_utilization()
            fetch_memory_utilization()
            time.sleep(3)

    import threading
    threading.Thread(target=fetch_metrics, daemon=True).start()

    # Start the Flask application
    app.run(host='127.0.0.1', port=4223)

