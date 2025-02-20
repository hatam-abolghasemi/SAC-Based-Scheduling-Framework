import time
import random
import requests
import json


job_generator_url = 'http://0.0.0.0:9902/jobs'
job_deploy_url = 'http://0.0.0.0:9901/deploy_job'
schedule_url = 'http://0.0.0.0:9902/schedule'
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
            print(f"Error accessing job generator service. Retrying in 5 seconds...")
        time.sleep(5)


def choose_best_node(job):
    node_name = f"k8s-worker-{random.randint(1, 13)}"
    return node_name


scheduled_generation_ids = set()
def schedule_jobs():
    while True:
        generated_jobs = fetch_generated_jobs()
        for job in generated_jobs:
            generation_id = job.get('generation_id')
            if generation_id in scheduled_generation_ids:
                print(f"Job with generation_id {generation_id} already scheduled.")
                continue
            node_name = choose_best_node(job)
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
                        print(f"Job with generation_id {generation_id} scheduled on node {node_name} successfully.")
                        scheduled_generation_ids.add(generation_id)
                        schedule_response = requests.post(schedule_url, json={'generation_id': generation_id})
                        if schedule_response.status_code != 200:
                            print(f"Failed to notify job-generator for job {job.get('job_id')}.")
                        break
                    else:
                        print(f"Failed to deploy job with generation_id {generation_id}. Response: {deploy_response.text}")
                except requests.RequestException as e:
                    print(f"Error accessing job deploy service. Retrying in 5 seconds...")
                time.sleep(5)
        time.sleep(5)


if __name__ == '__main__':
    schedule_jobs()

