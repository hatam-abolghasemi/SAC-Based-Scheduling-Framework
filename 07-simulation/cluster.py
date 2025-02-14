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
            "cpu": int(self.cpu),
            "memory": int(self.memory),
            "gpu": int(self.gpu)
        }

    def deploy_job(self, job):
        self.jobs.append(job)

# Worker nodes
nodes = [
    Node('worker', f'k8s-worker-{i+1}', '32', '96', '5120') for i in range(13)
]

node_dict = {node.name: node for node in nodes}
all_jobs = []

class ClusterHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/nodes':
            nodes_list = json.dumps([node.to_dict() for node in nodes], indent=2)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(nodes_list.encode())
        elif self.path == '/pods':
            if all_jobs:
                pods_list = '\n'.join(json.dumps(job) for job in all_jobs)
            else:
                pods_list = ""
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

    def do_POST(self):
        if self.path == '/deploy_job':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            job_data = json.loads(post_data.decode('utf-8'))
            
            # Extract job details from the request
            generation_id = job_data.get('generation_id')
            job_id = job_data.get('job_id')
            node_name = job_data.get('node')
            required_epoch = job_data.get('required_epoch')
            generation_moment = job_data.get('generation_moment')
            schedule_moment = job_data.get('schedule_moment')

            # Check if all required fields are present
            if not all([generation_id, job_id, node_name, required_epoch, generation_moment, schedule_moment]):
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Missing required fields"}).encode())
                return

            # Find the corresponding node and assign the job
            if node_name in node_dict:
                job = {
                    "generation_id": generation_id,
                    "job_id": job_id,
                    "node": node_name,
                    "required_epoch": required_epoch,
                    "generation_moment": generation_moment,
                    "schedule_moment": schedule_moment
                }
                all_jobs.append(job)
                node_dict[node_name].deploy_job(job)

                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Job deployed successfully"}).encode())
            else:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Invalid node"}).encode())

        elif self.path == '/schedule':
            # Notify that the job is scheduled (i.e., job generator has been informed)
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            schedule_data = json.loads(post_data.decode('utf-8'))
            
            generation_id = schedule_data.get('generation_id')

            # Check if generation_id is provided
            if not generation_id:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Missing generation_id"}).encode())
                return

            # Mark job as scheduled by removing it from the job list
            job_to_remove = None
            for job in all_jobs:
                if job['generation_id'] == generation_id:
                    job_to_remove = job
                    break

            if job_to_remove:
                all_jobs.remove(job_to_remove)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Job scheduled and removed from list"}).encode())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Job not found"}).encode())

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

