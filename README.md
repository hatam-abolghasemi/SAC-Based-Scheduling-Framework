State:
container_cpu_usage
container_mem_usage
container_gpu_usage
container_disk_io
container_net_io
node_cpu_utilization
node_mem_utilization
node_gpu_utilization
node_cpu_load1
node_container_count
dl_batch_size
dl_learning_rate
dl_expected_time
dl_requested_cpu
dl_requested_mem
dl_requested_gpu
cluster_node_count
cluster_queue_length

Action:
resource_allocation(allocated_cpu,allocated_mem,allocated_gpu)
node_selection(node_score,node_id)
hyperparameter_adaptation(batch_size,learning_rate)

Reward:
a1. faster job execution          -> 1                 / completion_tim
a2. optimal utilization           -> utilization       / requested
a3. penalize excessive allocation -> over_provisioning / total_capacity
a4. penalize excessive queueing   -> latency_penalty

r = a1 + a2 - a3 - a4

