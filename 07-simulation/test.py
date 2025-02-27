import sqlite3

# Connect to the jobs database
conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()

# Fetch all job details from the database
cursor.execute('SELECT job_id, job_batch_size, job_learning_rate, job_dataset_complexity, job_model_complexity FROM jobs WHERE job_id=9')

# Fetch all rows and print the job information
jobs = cursor.fetchall()
for job in jobs:
    job_id, job_batch_size, job_learning_rate, job_dataset_complexity, job_model_complexity = job
    print(f"job_id: {job_id}, job_batch_size: {job_batch_size}, job_learning_rate: {job_learning_rate}, "
          f"job_dataset_complexity: {job_dataset_complexity}, job_model_complexity: {job_model_complexity}")

# Close the connection to the database
conn.close()

