import sqlite3
import time
import logging
from flask import Flask, Response, request, jsonify
import threading
import subprocess
from datetime import datetime, timedelta

MIN_JOBS = 1
MAX_JOBS = 2
TOTAL_JOB_SAMPLES = 31
FLASK_PORT = 9902
JOB_EXPIRATION_TIME = 15
PATTERN_FILE = "/home/hatam/SAC-Based-Scheduling-Framework/inputs/pattern-train.txt"  # NEW

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.basicConfig(filename='/home/hatam/SAC-Based-Scheduling-Framework/generated_jobs.log', level=logging.INFO, format='%(asctime)s - %(message)s')

generated_jobs = []
job_queue = []
daily_job_count = 0
last_reset_date = datetime.now().date()
start_datetime = datetime.now()
start_time = time.time()
pattern_jobs = {}  # NEW
generation_counter = 1  # Tracks generation order

def load_job_pattern():  # NEW
    global pattern_jobs
    with open(PATTERN_FILE, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split(' ', 1)
            second = int(parts[0])
            job_str = parts[1]
            if second not in pattern_jobs:
                pattern_jobs[second] = []
            pattern_jobs[second].append(job_str)

def parse_job_string(job_str):  # UPDATED
    fields = job_str.split()
    job = {}
    for field in fields:
        key, value = field.split('=')
        if key == 'job_id':
            job['job_id'] = int(value)
        elif key == 'epochs':
            job['required_epoch'] = int(value)
    return job

def introduce_jobs():
    global generated_jobs, daily_job_count, last_reset_date, generation_counter
    while True:
        seconds_since_start = int(time.time() - start_time)
        now = start_datetime + timedelta(seconds=seconds_since_start)

        if now.date() != last_reset_date:
            daily_job_count = 0
            last_reset_date = now.date()
            logging.info("Daily job count reset.")

        hour = now.hour
        minute = now.minute
        activity_profile = [
            0.012, 0.012, 0.012, 0.012, 0.012,   # early morning
            0.025, 0.4, 0.05, 0.075, 0.075,     # working hours
            0.075, 0.1, 0.1, 0.075, 0.065,
            0.05, 0.04, 0.03, 0.03, 0.03,
            0.015, 0.08, 0.006, 0.004           # late night
        ]
        activity_level = activity_profile[hour]

        if daily_job_count < 80000 and seconds_since_start in pattern_jobs:
            result = subprocess.run(
                "ss -nlpt | grep 0.0.0.0:11 | grep python3 | wc -l",
                shell=True,
                capture_output=True,
                text=True
            )
            result_value = int(result.stdout.strip())
            if result_value < 810000:
                for job_str in pattern_jobs[seconds_since_start]:
                    if daily_job_count >= 800000:
                        break
                    job_info = parse_job_string(job_str)
                    job = {
                        'generation_id': generation_counter,
                        'generation_moment': int(time.time()),
                        'job_id': job_info['job_id'],
                        'required_epoch': job_info['required_epoch']
                    }
                    generated_jobs.append(job)
                    generation_counter += 1
                    daily_job_count += 1
                    logging.info(f"RealTime {hour:02d}:{minute:02d} - Job introduced: {job}")
            else:
                logging.info(f"RealTime {hour:02d}:{minute:02d} - Too many processes ({result_value}). Skipping.")
        else:
            logging.info(f"RealTime {hour:02d}:{minute:02d} - No job or job cap reached.")

        time.sleep(1)

def clean_generated_jobs():
    global generated_jobs
    while True:
        current_time = int(time.time())
        generated_jobs = [job for job in generated_jobs if current_time - job['generation_moment'] <= JOB_EXPIRATION_TIME]
        time.sleep(1)

@app.route('/jobs', methods=['GET'])
def get_jobs():
    return Response("".join([f"{str(job)}\n" for job in generated_jobs]), mimetype='text/plain')

@app.route('/queue', methods=['POST'])
def queue_job():
    generation_id = request.json.get('generation_id')
    global generated_jobs, job_queue
    job_to_queue = next((job for job in generated_jobs if job['generation_id'] == generation_id), None)
    if job_to_queue:
        job_queue.append(job_to_queue)
        logging.info(f"Job with generation_id {generation_id} added to the queue.")
        return jsonify({"message": f"Job with generation_id {generation_id} added to the queue."}), 200
    else:
        return jsonify({"error": "Job not found."}), 404

# NEW: Load the pattern on startup
load_job_pattern()

# Start background threads
thread_job_generation = threading.Thread(target=introduce_jobs)
thread_job_generation.daemon = True
thread_job_generation.start()

thread_generated_job_cleaning = threading.Thread(target=clean_generated_jobs)
thread_generated_job_cleaning.daemon = True
thread_generated_job_cleaning.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT)

