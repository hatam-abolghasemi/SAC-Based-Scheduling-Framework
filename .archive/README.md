State:
job_matrix=j*5
- job_id*(job_batch_size,job_required_epoch,job_learning_rate,job_dataset_complexity,job_model_complexity)
- this matrix contains static data about the properties of all running jobs
- everything is represented as numbers.
job_status=j*6
- job_id*(job_training_progress,job_training_accuracy,job_training_loss,job_cpu_usage,job_gpu_usage,job_mem_usage)
- this matrix container dynamic data about the current state of all jobs
node_status=n*4
- node_id*(node_cpu_utilization,node_gpu_utilization,node_mem_utilization,node_job_count)

Action:
node_selection()
- There must be a queue of jobs that need to be scheduled with a matrix like job_matrix.
- Scheduler selects a node for each queued job based on this matrix.

Reward:
- more average passed_epoch of one node, more reward
- more resource utilization metrics of a node, more reward
- less average elapsed time, more reward

