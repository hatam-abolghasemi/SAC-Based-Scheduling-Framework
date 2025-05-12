import os
import time
import requests
import subprocess
import threading
from datetime import datetime

processed_generation_ids = set()
lock = threading.Lock()
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_jobs(worker_id):
    url = f"http://localhost:9901/jobs/k8s-worker-{worker_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        jobs = response.json()
        return jobs
    except requests.RequestException as e:
        print(f"Error fetching jobs from worker {worker_id}: {e}")
        return []

def calculate_port(generation_id):
    if generation_id < 10:
        return f"1100{generation_id}"
    elif generation_id < 100:
        return f"110{generation_id}"
    elif generation_id < 1000:
        return f"11{generation_id}"
    elif generation_id < 2000:
        return f"2{generation_id}"
    elif generation_id < 3000:
        return f"3{generation_id}"
    elif generation_id < 4000:
        return f"4{generation_id}"
    elif generation_id < 5000:
        return f"5{generation_id}"
    elif generation_id < 5500:
        return f"6{generation_id}"
    else:
        raise ValueError(f"Generation ID {generation_id} too high for safe port assignment.")


def get_job_info(job_id, generation_id, jobs_data):
    for job in jobs_data:
        if job['job_id'] == job_id and job['generation_id'] == generation_id:
            return job
    return {}

def start_container_exporter(port, generation_id, job_info, job_id):
    image_name = f"job-exporter-{job_id}"
    container_name = f"job-{generation_id}"

    if not image_exists(image_name):
        print(f"Docker image {image_name} not found. Skipping...")
        return

    command = [
        'docker', 'run', '--rm', '-d', '--network', 'host', '--name', container_name,
        image_name,
        '--port', str(port),
        '--generation_id', str(generation_id),
        '--node', job_info.get('node', 'default-node'),
        '--job_id', str(job_info.get('job_id')),
        '--generation_moment', str(job_info.get('generation_moment', 0)),
        '--schedule_moment', str(job_info.get('schedule_moment', 0)),
        '--required_epochs', str(job_info.get('required_epoch', 0))
    ]
    print(f"[{timestamp}] [JobStart] job_id={job_id} gen_id={generation_id} port={port} node={job_info.get('node', 'default-node')} gen_moment={job_info.get('generation_moment', 0)} sched_moment={job_info.get('schedule_moment', 0)} epochs={job_info.get('required_epoch', 0)} image={image_name} container={container_name}", flush=True)
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def image_exists(image_name):
    try:
        subprocess.run(['docker', 'image', 'inspect', image_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def process_job(job, jobs):
    job_id = job.get("job_id")
    generation_id = int(job.get("generation_id", 0))
    if 1 <= job_id <= 33:
        with lock:  # Ensure thread-safe checks
            if generation_id not in processed_generation_ids:
                processed_generation_ids.add(generation_id)
                port = calculate_port(generation_id)
                job_info = get_job_info(job_id, generation_id, jobs)
                if job_info:
                    threading.Thread(target=start_container_exporter, args=(port, generation_id, job_info, job_id), daemon=True).start()

def monitor_jobs():
    print("Starting continuous job monitoring...")
    while True:
        for worker_id in range(1, 14):
            jobs = get_jobs(worker_id)
            for job in jobs:
                threading.Thread(target=process_job, args=(job, jobs), daemon=True).start()
        time.sleep(1)

if __name__ == '__main__':
    monitor_jobs()
