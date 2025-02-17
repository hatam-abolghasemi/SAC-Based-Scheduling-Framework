import requests
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import re

class NodeExporterHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            metrics = []

            # Step 1: Fetch node data from the cluster
            response = requests.get('http://0.0.0.0:9901/nodes')
            if response.status_code == 200:
                nodes = response.json()
                for node in nodes:
                    name = node['name']
                    metrics.append(f'node_total_mem{{name="{name}"}} {node["memory"]}')
                    metrics.append(f'node_total_cpu{{name="{name}"}} {node["cpu"]}')
                    metrics.append(f'node_total_gpu{{name="{name}"}} {node["gpu"]}')
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'Failed to fetch node data')
                return

            # Step 2: Find exporters and sum container_used_cpu, container_used_gpu, container_used_mem by node
            usage = {'cpu': {}, 'gpu': {}, 'mem': {}}
            try:
                result = subprocess.run(['ss', '-nlpt'], capture_output=True, text=True)
                ports = re.findall(r'0\.0\.0\.0:(11[0-9]+)', result.stdout)
                for port in ports:
                    try:
                        curl_response = subprocess.run(['curl', f'http://0.0.0.0:{port}/metrics'], capture_output=True, text=True)
                        for metric, key in [(r'^container_used_cpu.*node="([^"]+)".*\}\s+([0-9.]+)', 'cpu'),
                                            (r'^container_used_gpu.*node="([^"]+)".*\}\s+([0-9.]+)', 'gpu'),
                                            (r'^container_used_mem.*node="([^"]+)".*\}\s+([0-9.]+)', 'mem')]:
                            matches = re.findall(metric, curl_response.stdout, re.MULTILINE)
                            for node_name, value in matches:
                                usage[key][node_name] = usage[key].get(node_name, 0) + float(value)
                    except Exception as e:
                        print(f"Error accessing metrics on port {port}: {e}")

                # Report usage and free metrics with default 0
                for node in nodes:
                    name = node['name']
                    used_cpu = usage["cpu"].get(name, 0)
                    used_gpu = usage["gpu"].get(name, 0)
                    used_mem = usage["mem"].get(name, 0)

                    total_cpu = node["cpu"]
                    total_gpu = node["gpu"]
                    total_mem = node["memory"]

                    # Calculate free resources
                    free_cpu = total_cpu - used_cpu
                    free_gpu = total_gpu - used_gpu
                    free_mem = total_mem - used_mem

                    metrics.append(f'node_used_cpu{{name="{name}", node_epu_total="true"}} {used_cpu}')
                    metrics.append(f'node_used_gpu{{name="{name}", node_epu_total="true"}} {used_gpu}')
                    metrics.append(f'node_used_mem{{name="{name}", node_epu_total="true"}} {used_mem}')
                    
                    # Add free metrics
                    metrics.append(f'node_free_cpu{{name="{name}", node_epu_total="true"}} {free_cpu}')
                    metrics.append(f'node_free_gpu{{name="{name}", node_epu_total="true"}} {free_gpu}')
                    metrics.append(f'node_free_mem{{name="{name}", node_epu_total="true"}} {free_mem}')
            except Exception as e:
                print(f"Error during exporter scan: {e}")

            # Return collected metrics
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write('\n'.join(metrics).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 9904), NodeExporterHandler)
    print("Node exporter is running on port 9904...")
    try:
        while True:
            server.handle_request()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Node exporter...")
        server.server_close()

