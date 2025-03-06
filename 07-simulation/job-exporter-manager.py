import os
import time
import requests
import subprocess
import threading

processed_generation_ids = set()
lock = threading.Lock()

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
    else:
        return f"11{generation_id}"

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
    print(f"Launching container exporter for job {job_id} on port {port}...")
    subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def process_job(job, jobs):
    job_id = job.get("job_id")
    generation_id = int(job.get("generation_id", 0))
    if 1 <= job_id <= 16:
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

