import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

class NodeExporterHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Check the path of the incoming request
        if self.path == '/metrics':
            # Fetch node data from the cluster
            response = requests.get('http://0.0.0.0:9901/nodes')  # Adjust the cluster server URL if needed
            if response.status_code == 200:
                nodes = response.json()
                
                # Prepare metrics to expose
                metrics = []
                for node in nodes:
                    name = node["name"]
                    cpu = node["cpu"]
                    memory = node["memory"]
                    gpu = node["gpu"]
                    
                    # Add Prometheus-style metrics for each node
                    metrics.append(f'node_memory_total{{name="{name}"}} {memory}')
                    metrics.append(f'node_cpu_total{{name="{name}"}} {cpu}')
                    metrics.append(f'node_gpu_total{{name="{name}"}} {gpu}')
                
                # Return the metrics as a response
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write("\n".join(metrics).encode())
            else:
                # If the node data is not accessible, return an error
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write('{"status": "error", "message": "Failed to fetch node data"}'.encode())
        else:
            # For any other path, return 404
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    # Create a simple HTTP server to expose the metrics
    server = HTTPServer(('0.0.0.0', 9904), NodeExporterHandler)
    print("Node exporter is running on port 9904...")
    try:
        # Keep the server running indefinitely
        while True:
            server.handle_request()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Node exporter...")
        server.server_close()

