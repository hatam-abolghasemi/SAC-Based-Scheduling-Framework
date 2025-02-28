import requests
import subprocess
from prometheus_client import start_http_server, Gauge, REGISTRY
import time
import re
import sys


def fetch_node_data():
    connection_restored = False
    while True:
        try:
            response = requests.get('http://0.0.0.0:9901/nodes')
            if response.status_code == 200:
                if connection_restored:
                    sys.stderr.write("Connection re-established to cluster.\n")
                    connection_restored = False
                return response.json()
        except requests.exceptions.RequestException as e:
            sys.stderr.write(f"Failed to fetch data from cluster: {e}\nRetrying in 3 seconds...\n")
            connection_restored = True
        time.sleep(3)


def fetch_container_metrics(port):
    usage = {'cpu': {}, 'gpu': {}, 'mem': {}}
    try:
        curl_response = subprocess.run(['curl', f'http://0.0.0.0:{port}/metrics'], capture_output=True, text=True)
        for metric, key in [(r'^container_used_cpu.*node="([^"]+)".*\}\s+([0-9.]+)', 'cpu'),
                            (r'^container_used_gpu.*node="([^"]+)".*\}\s+([0-9.]+)', 'gpu'),
                            (r'^container_used_mem.*node="([^"]+)".*\}\s+([0-9.]+)', 'mem')]:
            matches = re.findall(metric, curl_response.stdout, re.MULTILINE)
            for node_name, value in matches:
                usage[key][node_name] = usage[key].get(node_name, 0) + float(value)
    except Exception as e:
        sys.stderr.write(f"Error accessing metrics on port {port}: {e}\n")
    return usage


def update_metrics():
    nodes = fetch_node_data()
    metrics = {
        'node_total_mem': Gauge('node_total_mem', 'Total memory of node', ['name']),
        'node_total_cpu': Gauge('node_total_cpu', 'Total CPU of node', ['name']),
        'node_total_gpu': Gauge('node_total_gpu', 'Total GPU of node', ['name']),
        'node_used_cpu': Gauge('node_used_cpu', 'Used CPU of node', ['name']),
        'node_used_gpu': Gauge('node_used_gpu', 'Used GPU of node', ['name']),
        'node_used_mem': Gauge('node_used_mem', 'Used memory of node', ['name']),
        'node_free_cpu': Gauge('node_free_cpu', 'Free CPU of node', ['name']),
        'node_free_gpu': Gauge('node_free_gpu', 'Free GPU of node', ['name']),
        'node_free_mem': Gauge('node_free_mem', 'Free memory of node', ['name']),
        'node_cpu_utilization': Gauge('node_cpu_utilization', 'CPU utilization of node', ['name']),
        'node_gpu_utilization': Gauge('node_gpu_utilization', 'GPU utilization of node', ['name']),
        'node_mem_utilization': Gauge('node_mem_utilization', 'Memory utilization of node', ['name']),
    }
    while True:
        try:
            result = subprocess.run(['ss', '-nlpt'], capture_output=True, text=True)
            ports = re.findall(r'0\.0\.0\.0:(11[0-9]+)', result.stdout)
            usage = {'cpu': {}, 'gpu': {}, 'mem': {}}
            for port in ports:
                container_usage = fetch_container_metrics(port)
                for key in usage:
                    for node, value in container_usage[key].items():
                        usage[key][node] = usage[key].get(node, 0) + value
            for node in nodes:
                name = node['name']
                total_cpu = node['cpu']
                total_gpu = node['gpu']
                total_mem = node['memory']
                used_cpu = usage['cpu'].get(name, 0)
                used_gpu = usage['gpu'].get(name, 0)
                used_mem = usage['mem'].get(name, 0)
                free_cpu = total_cpu - used_cpu
                free_gpu = total_gpu - used_gpu
                free_mem = total_mem - used_mem
                cpu_utilization = int((used_cpu / total_cpu) * 100) if total_cpu > 0 else 0
                gpu_utilization = int((used_gpu / total_gpu) * 100) if total_gpu > 0 else 0
                mem_utilization = int((used_mem / total_mem) * 100) if total_mem > 0 else 0
                metrics['node_total_mem'].labels(name=name).set(total_mem)
                metrics['node_total_cpu'].labels(name=name).set(total_cpu)
                metrics['node_total_gpu'].labels(name=name).set(total_gpu)
                metrics['node_used_cpu'].labels(name=name).set(used_cpu)
                metrics['node_used_gpu'].labels(name=name).set(used_gpu)
                metrics['node_used_mem'].labels(name=name).set(used_mem)
                metrics['node_free_cpu'].labels(name=name).set(free_cpu)
                metrics['node_free_gpu'].labels(name=name).set(free_gpu)
                metrics['node_free_mem'].labels(name=name).set(free_mem)
                metrics['node_cpu_utilization'].labels(name=name).set(cpu_utilization)
                metrics['node_gpu_utilization'].labels(name=name).set(gpu_utilization)
                metrics['node_mem_utilization'].labels(name=name).set(mem_utilization)
            time.sleep(1)
        except Exception as e:
            sys.stderr.write(f"Error during metric update: {e}\n")
            time.sleep(3)


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


if __name__ == '__main__':
    deregister_unwanted_metrics()
    start_http_server(9904)
    sys.stderr.write("Node exporter is running on port 9904...\n")
    update_metrics()

