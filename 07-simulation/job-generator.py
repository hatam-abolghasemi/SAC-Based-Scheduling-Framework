import sqlite3
import random
import time
import logging
from flask import Flask, Response, request, jsonify
import threading

JOB_INTRO_MIN_SECONDS = 15
JOB_INTRO_MAX_SECONDS = 60
MIN_JOBS = 1
MAX_JOBS = 4
TOTAL_JOB_SAMPLES = 16
FLASK_PORT = 9902
MIN_REQUIRED_EPOCH = 10
MAX_REQUIRED_EPOCH = 500
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.basicConfig(filename='generated_jobs.log', level=logging.INFO, format='%(asctime)s - %(message)s')
generated_jobs = []
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
        jobs = generate_jobs()
        for job in jobs:
            generated_jobs.append(job)
            logging.info(f"Job introduced: {job}")
        time.sleep(random.randint(JOB_INTRO_MIN_SECONDS, JOB_INTRO_MAX_SECONDS))


@app.route('/jobs', methods=['GET'])
def get_jobs():
    return Response("".join([f"{str(job)}\n" for job in generated_jobs]), mimetype='text/plain')


@app.route('/schedule', methods=['POST'])
def schedule_job():
    generation_id = request.json.get('generation_id')
    global generated_jobs
    generated_jobs = [job for job in generated_jobs if job['generation_id'] != generation_id]
    return jsonify({"message": f"Job with generation_id {generation_id} is scheduled and removed from the list."}), 200


thread = threading.Thread(target=introduce_jobs)
thread.daemon = True
thread.start()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT)

