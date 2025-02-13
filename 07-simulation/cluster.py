import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import argparse

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
            "type": self.node_type,
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu,
            "jobs": [job['job_id'] for job in self.jobs]  # Only returning job IDs for simplicity
        }

    def deploy_job(self, job):
        self.jobs.append(job)

nodes = [
    Node('master', f'k8s-master-{i+1}', '8 core', '16GB', 'None') for i in range(5)
] + [
    Node('worker', f'k8s-worker-{i+1}', '32 core', '96GB', 'NVIDIA DGX A100') for i in range(13)
]

node_dict = {node.name: node for node in nodes}
all_jobs = []

class ClusterHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/nodes':
            # Return a list of all nodes
            nodes_list = '\n'.join(node.name for node in nodes)
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(nodes_list.encode())
        elif self.path == '/pods':
            # Return list of all jobs (pods)
            if all_jobs:
                pods_list = '\n'.join(json.dumps(job) for job in all_jobs)
            else:
                pods_list = ""
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(pods_list.encode())
        elif self.path == '/jobs':
            # Provide a list of jobs currently scheduled to nodes
            jobs_list = json.dumps(all_jobs)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(jobs_list.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        # Handle job deployment to the node
        if self.path == '/deploy_job':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            job_data = json.loads(post_data.decode('utf-8'))
            job_id = job_data.get('job_id')
            node_name = job_data.get('node')

            # Find the corresponding node and assign the job
            if job_id and node_name in node_dict:
                job = {"job_id": job_id, "node": node_name}
                all_jobs.append(job)
                node_dict[node_name].deploy_job(job)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Job deployed successfully"}).encode())
            else:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Invalid job or node"}).encode())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9901, help='Port to expose cluster endpoints')
    args = parser.parse_args()

    server = HTTPServer(('0.0.0.0', args.port), ClusterHandler)
    print("Cluster is ready. Waiting for job deployments...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down cluster...")
        server.server_close()

