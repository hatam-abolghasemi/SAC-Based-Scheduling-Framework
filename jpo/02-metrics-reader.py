import requests
import time

PROMETHEUS_URL = 'http://127.0.0.1:9090'
CPU_UTILIZATION_QUERY = 'avg(100 - (avg by(instance) (irate(node_cpu_seconds_total{job="local_node_exporter", mode="idle"}[5m])) * 100))'
MEMORY_UTILIZATION_QUERY = 'avg((1 - (node_memory_MemAvailable_bytes{job="local_node_exporter"} / node_memory_MemTotal_bytes{job="local_node_exporter"})) * 100)'

def fetch_cpu_utilization():
    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': CPU_UTILIZATION_QUERY})
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            results = data['data']['result']
            if results:
                value = float(results[0]['value'][1])
                unix_time = int(time.time())
                print(f'{unix_time} cpu_utilization {value:.2f}%')

def fetch_memory_utilization():
    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': MEMORY_UTILIZATION_QUERY})
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            results = data['data']['result']
            if results:
                value = float(results[0]['value'][1])
                unix_time = int(time.time())
                print(f'{unix_time} memory_utilization {value:.2f}%')

if __name__ == "__main__":
    while True:
        fetch_cpu_utilization()
        fetch_memory_utilization()
        time.sleep(15)

