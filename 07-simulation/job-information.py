import sqlite3

# Connect to the jobs database
conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()

# Fetch all job details from the database
cursor.execute('SELECT job_id, dl_batch_size, dl_learning_rate, dl_dataset, dl_framework, dl_model FROM jobs')

# Fetch all rows and print the job information
jobs = cursor.fetchall()
for job in jobs:
    job_id, dl_batch_size, dl_learning_rate, dl_dataset, dl_framework, dl_model = job
    print(f"job_id: {job_id}, dl_batch_size: {dl_batch_size}, dl_learning_rate: {dl_learning_rate}, "
          f"dl_dataset: {dl_dataset}, dl_framework: {dl_framework}, dl_model: {dl_model}")

# Close the connection to the database
conn.close()

