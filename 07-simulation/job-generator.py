import sqlite3
import random
import time
import logging
from flask import Flask, Response, request, jsonify
import threading

app = Flask(__name__)

# Disable Flask's default logging (requests, etc.)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Setup logging to a file for generated jobs
logging.basicConfig(filename='generated_jobs.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Global variables
generated_jobs = []
generation_counter = 1  # Counter for generation_id
start_time = time.time()  # Track the start time for elapsed seconds

# Function to get a random selection of jobs from the database
def generate_jobs():
    global generation_counter
    # Connect to the SQLite database
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()

    # Randomly choose between 1 and 4 job_ids
    num_jobs = random.randint(1, 4)

    # Select random job IDs from 1 to 16
    job_ids = random.sample(range(1, 17), num_jobs)

    jobs = []
    for job_id in job_ids:
        cursor.execute('''
        SELECT dl_batch_size, dl_learning_rate, dl_expected_time, dl_requested_cpu, dl_requested_mem, dl_requested_gpu, dl_dataset, dl_framework, dl_model
        FROM jobs WHERE job_id = ?
        ''', (job_id,))
        job = cursor.fetchone()

        # Calculate elapsed time in seconds
        elapsed_time = int(time.time() - start_time)

        job_dict = {
            'generation_id': generation_counter,  # Add generation_id
            'elapsed_time': elapsed_time,  # Add elapsed time
            'job_id': job_id,
            'dl_batch_size': job[0],
            'dl_learning_rate': job[1],
            'dl_expected_time': job[2],
            'dl_requested_cpu': job[3],
            'dl_requested_mem': job[4],
            'dl_requested_gpu': job[5],
            'dl_dataset': job[6],
            'dl_framework': job[7],
            'dl_model': job[8]
        }
        jobs.append(job_dict)
        generation_counter += 1  # Increment the generation_id

    conn.close()
    return jobs

# Function to introduce jobs periodically
def introduce_jobs():
    global generated_jobs
    while True:
        jobs = generate_jobs()
        for job in jobs:
            generated_jobs.append(job)  # Append each new job to the list one at a time
            logging.info(f"Job introduced: {job}")  # Log the job to the file instead of printing to stdout
        time.sleep(random.randint(1, 40))  # Introduce jobs periodically every 1 to 40 seconds

# API endpoint to serve the generated jobs in the desired format
@app.route('/jobs', methods=['GET'])
def get_jobs():
    # Create the plain text response with jobs printed on separate lines
    job_text = ""
    for job in generated_jobs:
        job_text += f"Elapsed Time: {job['elapsed_time']}s | Generation ID: {job['generation_id']} | {str(job)}\n"
    return Response(job_text, mimetype='text/plain')

# API endpoint to receive scheduling notification
@app.route('/schedule', methods=['POST'])
def schedule_job():
    data = request.json
    generation_id = data.get('generation_id')

    # Remove the job with the provided generation_id from the list
    global generated_jobs
    generated_jobs = [job for job in generated_jobs if job['generation_id'] != generation_id]

    return jsonify({"message": f"Job with generation_id {generation_id} is scheduled and removed from the list."}), 200

# Start the job generation in a separate thread
thread = threading.Thread(target=introduce_jobs)
thread.daemon = True
thread.start()

# Run the Flask app to serve the jobs at 0.0.0.0:9902
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9902)

