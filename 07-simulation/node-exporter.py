import requests
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import re
import sys


class NodeExporterHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to suppress standard logging of requests."""
        return

    def do_GET(self):
        if self.path == '/metrics':
            metrics = []
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
                        sys.stderr.write(f"Error accessing metrics on port {port}: {e}\n")
                for node in nodes:
                    name = node['name']
                    used_cpu = usage["cpu"].get(name, 0)
                    used_gpu = usage["gpu"].get(name, 0)
                    used_mem = usage["mem"].get(name, 0)
                    total_cpu = node["cpu"]
                    total_gpu = node["gpu"]
                    total_mem = node["memory"]
                    free_cpu = total_cpu - used_cpu
                    free_gpu = total_gpu - used_gpu
                    free_mem = total_mem - used_mem
                    metrics.append(f'node_used_cpu{{name="{name}"}} {used_cpu}')
                    metrics.append(f'node_used_gpu{{name="{name}"}} {used_gpu}')
                    metrics.append(f'node_used_mem{{name="{name}"}} {used_mem}')
                    metrics.append(f'node_free_cpu{{name="{name}"}} {free_cpu}')
                    metrics.append(f'node_free_gpu{{name="{name}"}} {free_gpu}')
                    metrics.append(f'node_free_mem{{name="{name}"}} {free_mem}')
            except Exception as e:
                sys.stderr.write(f"Error during exporter scan: {e}\n")
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write('\n'.join(metrics).encode())
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 9904), NodeExporterHandler)
    sys.stderr.write("Node exporter is running on port 9904...\n")
    try:
        while True:
            server.handle_request()
            time.sleep(1)
    except KeyboardInterrupt:
        sys.stderr.write("\nShutting down Node exporter...\n")
        server.server_close()
