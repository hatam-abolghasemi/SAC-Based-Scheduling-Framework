import time
import requests
import json
import numpy as np
from deepLearningEnvironment import deepLearningEnvironment

job_generator_url = 'http://0.0.0.0:9902/jobs'
job_deploy_url = 'http://0.0.0.0:9901/deploy_job'
queue_url = 'http://0.0.0.0:9902/queue'

# Initialize environment
env = deepLearningEnvironment()

# Track scheduled jobs
scheduled_generation_ids = set()

def fetch_generated_jobs():
    while True:
        try:
            response = requests.get(job_generator_url)
            if response.status_code == 200:
                raw_response = response.text
                jobs = []
                for line in raw_response.splitlines():
                    try:
                        job = json.loads(line.replace("'", '"'))
                        jobs.append(job)
                    except json.JSONDecodeError as e:
                        print(f"Failed to decode line: {line}")
                        print(f"Error: {e}")
                return jobs
            else:
                print(f"Failed to fetch jobs, status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error accessing job generator service. Retrying in 1 second...")
        time.sleep(1)

def choose_best_node(job_queue, num_nodes):
    env.queued_jobs = job_queue
    env.update_action_space(num_nodes)
    action = np.random.uniform(1, num_nodes + 1, size=(len(job_queue),))
    node_assignments = np.round(np.clip(action, 1, num_nodes)).astype(int)
    return node_assignments

def schedule_jobs():
    while True:
        generated_jobs = fetch_generated_jobs()
        job_queue = [job['generation_id'] for job in generated_jobs if job['generation_id'] not in scheduled_generation_ids]
        num_nodes = 13  # Can be updated dynamically based on environment
        node_assignments = choose_best_node(job_queue, num_nodes)
        
        for job, node in zip(generated_jobs, node_assignments):
            generation_id = job.get('generation_id')
            if generation_id in scheduled_generation_ids:
                continue
            node_name = f"k8s-worker-{node}"
            schedule_moment = job.get('generation_moment')
            job_data = {
                'generation_id': generation_id,
                'job_id': job.get('job_id'),
                'node': node_name,
                'required_epoch': job.get('required_epoch'),
                'generation_moment': job.get('generation_moment'),
                'schedule_moment': schedule_moment
            }
            while True:
                try:
                    deploy_response = requests.post(job_deploy_url, json=job_data)
                    if deploy_response.status_code == 200:
                        print(f"Job {generation_id} scheduled on node {node_name}.")
                        scheduled_generation_ids.add(generation_id)
                        queue_response = requests.post(queue_url, json={'generation_id': generation_id})
                        if queue_response.status_code != 200:
                            print(f"Failed to notify queue for job {job.get('job_id')}.")
                        break
                    else:
                        print(f"Failed to deploy job {generation_id}. Response: {deploy_response.text}")
                except requests.RequestException as e:
                    print(f"Error accessing cluster. Retrying in 1 second...")
                time.sleep(1)
        time.sleep(60)

if __name__ == '__main__':
    schedule_jobs()

