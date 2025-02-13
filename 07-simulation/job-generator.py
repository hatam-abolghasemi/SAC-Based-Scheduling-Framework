import sqlite3
import random
import time
from flask import Flask, Response

app = Flask(__name__)

# Global variable to store the generated jobs
generated_jobs = []

# Function to get a random selection of jobs from the database
def generate_jobs():
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
        job_dict = {
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

    conn.close()
    return jobs

# Function to introduce jobs periodically
def introduce_jobs():
    global generated_jobs
    while True:
        jobs = generate_jobs()
        for job in jobs:
            generated_jobs.append(job)  # Append each new job to the list one at a time
            print(f"Job introduced: {job}")  # Print the job on a new line
        time.sleep(random.randint(1, 40))  # Introduce jobs periodically every 1 to 40 seconds

# API endpoint to serve the generated jobs in the desired format
@app.route('/jobs', methods=['GET'])
def get_jobs():
    # Create the plain text response with jobs printed on separate lines
    job_text = ""
    for job in generated_jobs:
        job_text += str(job) + "\n"  # Print each job in the dictionary-like format on a new line
    return Response(job_text, mimetype='text/plain')

# Start the job generation in a separate thread
import threading
thread = threading.Thread(target=introduce_jobs)
thread.daemon = True
thread.start()

# Run the Flask app to serve the jobs at 0.0.0.0:9902
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9902)

