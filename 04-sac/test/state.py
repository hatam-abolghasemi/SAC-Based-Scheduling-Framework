import requests
import numpy as np

class SchedulerState:
    def __init__(self, prometheus_url):
        self.prometheus_url = prometheus_url
    
    def query_prometheus(self, query):
        """Fetch data from Prometheus"""
        response = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query})
        if response.status_code == 200:
            result = response.json()["data"]["result"]
            value = float(result[0]["value"][1]) if result else 0.0
            print(f"Query: {query} -> Value: {value}")  # Logging query results
            return value
        print(f"Query failed: {query}")  # Logging failed queries
        return 0.0
    
    def get_state(self):
        """Construct the state vector"""
        state = {
            # Container-level metrics
            "container_cpu_usage": self.query_prometheus("sum(rate(container_cpu_usage_seconds_total[1m]))"),
            "container_mem_usage": self.query_prometheus("sum(container_memory_usage_bytes)"),
            "container_gpu_usage": self.query_prometheus("sum(container_gpu_memory_utilization)"),
            "container_disk_io": self.query_prometheus("sum(rate(container_fs_io_time_seconds_total[1m]))"),
            "container_net_io": self.query_prometheus("sum(rate(container_network_transmit_bytes_total[1m]))"),
            
            # Node-level metrics
            "node_cpu_utilization": self.query_prometheus("avg(rate(node_cpu_seconds_total[1m]))"),
            "node_mem_utilization": self.query_prometheus("avg(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))"),
            "node_gpu_utilization": self.query_prometheus("avg(node_gpu_utilization)"),
            "node_cpu_load1": self.query_prometheus("avg(node_load1)"),
            "node_container_count": self.query_prometheus("count(container_memory_usage_bytes)"),
            
            # Job-specific parameters (to be fetched from job metadata)
            "dl_batch_size": 32,  # Placeholder
            "dl_learning_rate": 0.001,  # Placeholder
            "dl_expected_time": self.query_prometheus("avg(job_expected_completion_time)"),
            "dl_requested_cpu": self.query_prometheus("sum(job_requested_cpu)"),
            "dl_requested_mem": self.query_prometheus("sum(job_requested_mem)"),
            "dl_requested_gpu": self.query_prometheus("sum(job_requested_gpu)"),
            
            # Cluster-level metrics
            "cluster_node_count": self.query_prometheus("count(node_cpu_seconds_total)"),
            "cluster_queue_length": self.query_prometheus("count(pending_jobs)"),
        }
        
        print("State Values:")  # Logging the state values
        for key, value in state.items():
            print(f"{key}: {value}")
        
        return np.array(list(state.values()), dtype=np.float32)

# Example usage
prometheus_url = "http://localhost:9090"
scheduler_state = SchedulerState(prometheus_url)
state_vector = scheduler_state.get_state()
print("State Vector:", state_vector)

