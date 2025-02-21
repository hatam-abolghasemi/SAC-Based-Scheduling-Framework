State:
job_training_progress
job_training_accuracy
job_training_loss
container_cpu_usage
container_mem_usage
container_gpu_usage
node_cpu_utilization
node_mem_utilization
node_gpu_utilization

Action:
node_selection(node_score,node_name)

Reward:
a1. faster job execution          -> max(job_progress)
a2. optimal utilization           -> max(node_utilization)

r = a1 + a2

