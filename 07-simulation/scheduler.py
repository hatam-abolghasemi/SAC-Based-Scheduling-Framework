import time
import random
import requests
import json

# Global variables
job_generator_url = 'http://0.0.0.0:9902/jobs'
job_deploy_url = 'http://0.0.0.0:9901/deploy_job'
schedule_url = 'http://0.0.0.0:9902/schedule'

# Fetch generated jobs
def fetch_generated_jobs():
    response = requests.get(job_generator_url)
    if response.status_code == 200:
        raw_response = response.text
        jobs = []
        for line in raw_response.splitlines():
            try:
                job = json.loads(line.replace("'", '"'))  # Convert single quotes to double quotes
                jobs.append(job)
            except json.JSONDecodeError as e:
                print(f"Failed to decode line: {line}")
                print(f"Error: {e}")
        return jobs
    else:
        print(f"Failed to fetch jobs, status code: {response.status_code}")
        return []

# Choose the best node (mock-up function for example)
def choose_best_node(job):
    # Modify to select a node name in the correct format
    node_name = f"k8s-worker-{random.randint(1, 13)}"  # Pick a random node from 1 to 13
    return node_name

# Schedule jobs
scheduled_generation_ids = set()

def schedule_jobs():
    while True:
        generated_jobs = fetch_generated_jobs()

        for job in generated_jobs:
            generation_id = job.get('generation_id')

            if generation_id in scheduled_generation_ids:
                print(f"Job with generation_id {generation_id} already scheduled.")
                continue

            job_id = job.get('job_id')  
            required_epoch = job.get('required_epoch')
            generation_moment = job.get('generation_moment')

            # Start the artificial time counter
            counter = 0

            # Logic for choosing the best node
            node_name = choose_best_node(job)  

            # Stop the counter once the best node is chosen
            schedule_moment = generation_moment + counter  # Add the counter value to generation_moment
            # Add schedule_moment to the job data
            job_data = {
                'generation_id': generation_id,  
                'job_id': job_id,
                'node': node_name,
                'required_epoch': required_epoch,
                'generation_moment': generation_moment,  # Keep the generation_moment
                'schedule_moment': schedule_moment     # Add schedule_moment
            }

            # Send POST request to deploy the job
            deploy_response = requests.post(job_deploy_url, json=job_data)

            if deploy_response.status_code == 200:
                print(f"Job with generation_id {generation_id} scheduled on node {node_name} successfully.")
                scheduled_generation_ids.add(generation_id)

                # Notify job generator that the job has been scheduled
                schedule_response = requests.post(schedule_url, json={'generation_id': generation_id})
                if schedule_response.status_code == 200:
                    pass
                else:
                    print(f"Failed to notify job-generator for job {job_id}.")
            else:
                print(f"Failed to deploy job with generation_id {generation_id} on node {node_name}. Response: {deploy_response.text}")

        # Sleep for a moment before checking again
        time.sleep(5)

if __name__ == '__main__':
    schedule_jobs()

