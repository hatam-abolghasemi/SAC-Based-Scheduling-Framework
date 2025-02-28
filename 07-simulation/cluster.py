import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import argparse
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class Node:
    def __init__(self, node_type, name, cpu, memory, gpu):
        self.name = name
        self.node_type = node_type
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu
        self.jobs = []


    def to_dict(self):
        return {
            "name": self.name,
            "cpu": int(self.cpu),
            "memory": int(self.memory),
            "gpu": int(self.gpu)
        }


    def deploy_job(self, job):
        self.jobs.append(job)


nodes = [
    Node('worker', f'k8s-worker-{i+1}', '32', '96', '5120') for i in range(13)
]
node_dict = {node.name: node for node in nodes}
all_jobs = []
class ClusterHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        try:
            if self.path == '/nodes':
                nodes_list = json.dumps([node.to_dict() for node in nodes], indent=2)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(nodes_list.encode())
            elif self.path.startswith('/jobs/'):
                node_name = self.path[len('/jobs/'):]
                if node_name in node_dict:
                    jobs_list = json.dumps(node_dict[node_name].jobs, indent=2)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(jobs_list.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            elif self.path == '/pods':
                pods_list = '\n'.join(json.dumps(job) for job in all_jobs) if all_jobs else ""
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(pods_list.encode())
            elif self.path == '/jobs':
                jobs_list = json.dumps(all_jobs, indent=2)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(jobs_list.encode())
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            logging.error(f"GET request error: {e}")
            self.send_response(500)
            self.end_headers()


    def do_POST(self):
        try:
            if self.path == '/deploy_job':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                job_data = json.loads(post_data.decode('utf-8'))
                node_name = job_data.get('node')
                if node_name in node_dict:
                    node_dict[node_name].deploy_job(job_data)
                    all_jobs.append(job_data)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "message": "Job deployed successfully"}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            else:
                self.send_response(501)
                self.end_headers()
        except Exception as e:
            logging.error(f"POST request error: {e}")
            self.send_response(500)
            self.end_headers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9901, help='Port to expose cluster endpoints')
    args = parser.parse_args()
    server = HTTPServer(('0.0.0.0', args.port), ClusterHandler)
    print("Cluster is running on port 9901...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down cluster...")
        server.server_close()

