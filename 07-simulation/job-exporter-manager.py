import os
import time
import requests
import subprocess
import threading


processed_generation_ids = set()
def get_jobs(worker_id):
    url = f"http://localhost:9901/jobs/k8s-worker-{worker_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        jobs = response.json()
        print(f"Fetched {len(jobs)} jobs from worker {worker_id}.")
        return jobs
    except requests.RequestException as e:
        print(f"Error fetching jobs from worker {worker_id}: {e}")
        return []


def calculate_port(generation_id):
    if generation_id < 10:
        port = "1100" + str(generation_id)
    elif generation_id < 100:
        port = "110" + str(generation_id)
    else:
        port = "11" + str(generation_id)
    return port


def get_job_info(job_id, generation_id, jobs_data):
    for job in jobs_data:
        if job['job_id'] == job_id and job['generation_id'] == generation_id:
            return job
    return {}


def start_container_exporter(port, generation_id, job_info, job_id):
    exporter_script = f'job-exporter-{job_id}.py'
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
    job_id = job.get("job_id")
    generation_id = int(job.get("generation_id", 0))
    if 1 <= job_id <= 16 and generation_id not in processed_generation_ids:
        port = calculate_port(generation_id)
        job_info = get_job_info(job_id, generation_id, jobs)
        if job_info:
            start_container_exporter(port, generation_id, job_info, job_id)
            processed_generation_ids.add(generation_id)


def monitor_jobs():
    print("Starting job monitoring loop...")
    while True:
        threads = []
        for worker_id in range(1, 14):
            jobs = get_jobs(worker_id)
            for job in jobs:
                thread = threading.Thread(target=process_job, args=(job, jobs))
                threads.append(thread)
                thread.start()
        for thread in threads:
            thread.join()
        print("Waiting before fetching jobs again...")
        time.sleep(10)


if __name__ == '__main__':
    monitor_jobs()

