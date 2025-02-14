import os
import time
import requests
import subprocess
import threading

# A set to track the processed generation_ids
processed_generation_ids = set()

def get_jobs():
    """GET request to check for jobs"""
    try:
        print("Fetching jobs from http://localhost:9901/jobs")
        response = requests.get("http://localhost:9901/jobs")
        response.raise_for_status()  # Raise error for bad status codes
        jobs = response.json()
        print(f"Fetched {len(jobs)} jobs.")
        return jobs  # Assuming the response is JSON
    except requests.RequestException as e:
        print(f"Error fetching jobs: {e}")
        return []

def calculate_port(generation_id):
    """Calculate port based on the generation_id"""
    if generation_id < 10:
        port = "1100" + str(generation_id)
    elif generation_id < 100:
        port = "110" + str(generation_id)
    else:
        port = "11" + str(generation_id)
    print(f"Calculated port {port} for generation_id {generation_id}")
    return port

def get_job_info(job_id, generation_id, jobs_data):
    """Fetch job details for a given job_id and generation_id from the list of jobs"""
    print(f"Searching for job with job_id {job_id} and generation_id {generation_id}")
    for job in jobs_data:
        if job['job_id'] == job_id and job['generation_id'] == generation_id:
            print(f"Found job with job_id {job_id} and generation_id {generation_id}")
            return job
    print(f"Job with job_id {job_id} and generation_id {generation_id} not found.")
    return {}

def start_container_exporter(port, generation_id, job_info, job_id):
    """Start the container exporter for a specific job_id with job details"""
    print(f"Starting container exporter for generation_id {generation_id} on port {port} for job_id {job_id}")
    exporter_script = f'job-exporter-{job_id}.py'
    
    # Check if exporter script exists for the job_id
    if not os.path.exists(exporter_script):
        print(f"Exporter script {exporter_script} not found. Skipping...")
        return
    
    command = [
        'python3', exporter_script,
        '--port', str(port),
        '--generation_id', str(generation_id),
        '--node', job_info.get('node', 'default-node'),
        '--job_id', str(job_info.get('job_id')),
        '--generation_moment', str(job_info.get('generation_moment', 0)),
        '--schedule_moment', str(job_info.get('schedule_moment', 0)),
        '--required_epochs', str(job_info.get('required_epoch', 0))
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    print("Command output:", result.stdout)
    if result.stderr:
        print("Command error:", result.stderr)

def process_job(job, jobs):
    """Process a single job in a separate thread"""
    job_id = job.get("job_id")
    generation_id = int(job.get("generation_id", 0))

    # Only process jobs with job_id between 1 and 16, and not already processed
    if 1 <= job_id <= 16 and generation_id not in processed_generation_ids:
        print(f"Processing job with job_id {job_id} and generation_id {generation_id}")
        port = calculate_port(generation_id)
        job_info = get_job_info(job_id, generation_id, jobs)
        if job_info:
            start_container_exporter(port, generation_id, job_info, job_id)
            processed_generation_ids.add(generation_id)  # Mark this generation_id as processed
        else:
            print(f"Skipping job with job_id {job_id} and generation_id {generation_id} due to missing info.")
    else:
        print(f"Skipping job with job_id {job_id} and generation_id {generation_id} (already processed or not valid job_id).")

def monitor_jobs():
    """Monitor jobs and start appropriate job exporter based on job_id"""
    print("Starting job monitoring loop...")
    while True:
        jobs = get_jobs()
        threads = []
        for job in jobs:
            thread = threading.Thread(target=process_job, args=(job, jobs))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish before checking for new jobs
        for thread in threads:
            thread.join()

        # Wait before checking for jobs again
        print("Waiting before fetching jobs again...")
        time.sleep(10)  # Adjust the delay as needed

if __name__ == '__main__':
    monitor_jobs()

