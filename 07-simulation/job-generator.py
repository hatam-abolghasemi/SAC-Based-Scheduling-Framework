import sqlite3
import random
import time
import logging
from flask import Flask, Response, request, jsonify
import threading
import subprocess

JOB_INTRO_MIN_SECONDS = 15
JOB_INTRO_MAX_SECONDS = 60
MIN_JOBS = 1
MAX_JOBS = 4
TOTAL_JOB_SAMPLES = 16
FLASK_PORT = 9902
MIN_REQUIRED_EPOCH = 10
MAX_REQUIRED_EPOCH = 500
JOB_EXPIRATION_TIME = 15
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.basicConfig(filename='generated_jobs.log', level=logging.INFO, format='%(asctime)s - %(message)s')
generated_jobs = []
job_queue = []
generation_counter = 1
start_time = time.time()

def generate_jobs():
    global generation_counter
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()
    num_jobs = random.randint(MIN_JOBS, MAX_JOBS)
    job_ids = random.sample(range(1, TOTAL_JOB_SAMPLES + 1), num_jobs)
    jobs = []
    for job_id in job_ids:
        cursor.execute('SELECT job_id FROM jobs WHERE job_id = ?', (job_id,))
        job = cursor.fetchone()
        generation_moment = max(int(time.time() - start_time), 1)
        required_epoch = random.choice(range(MIN_REQUIRED_EPOCH, MAX_REQUIRED_EPOCH + 1, 10))
        jobs.append({
            'generation_id': generation_counter,
            'generation_moment': generation_moment,
            'job_id': job_id,
            'required_epoch': required_epoch
        })
        generation_counter += 1
    conn.close()
    return jobs

def introduce_jobs():
    global generated_jobs
    while True:
        result = subprocess.run(
            "ss -nlpt | grep 0.0.0.0:11 | grep python3 | wc -l", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        result_value = int(result.stdout.strip())  # Get the result as an integer
        if result_value < 41:
            jobs = generate_jobs()
            for job in jobs:
                generated_jobs.append(job)
                logging.info(f"Job introduced: {job}")
        else:
            logging.info(f"Command result {result_value} is greater than or equal to 41. Skipping job generation.")
        time.sleep(random.randint(JOB_INTRO_MIN_SECONDS, JOB_INTRO_MAX_SECONDS))

def clean_generated_jobs():
    global generated_jobs
    while True:
        current_time = int(time.time() - start_time)
        generated_jobs = [job for job in generated_jobs if current_time - job['generation_moment'] <= JOB_EXPIRATION_TIME]
        time.sleep(1)  # Check every second

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

thread_job_generation = threading.Thread(target=introduce_jobs)
thread_job_generation.daemon = True
thread_job_generation.start()
thread_generated_job_cleaning = threading.Thread(target=clean_generated_jobs)
thread_generated_job_cleaning.daemon = True
thread_generated_job_cleaning.start()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT)

