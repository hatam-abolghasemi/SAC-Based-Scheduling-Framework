class SchedulingEnvironment:
    def __init__(self):
        self.job_data = None
        self.nodes = None

    def reset(self):
        self.state = self.get_state()
        self.job_data = self.get_job_data()
        self.nodes = self.get_nodes()
        return self.state

    def fetch_state(self):
        try:
            # Fetch the state from the API
            response = requests.get("http://0.0.0.0:9907/state")
            if response.status_code == 200:
                return response.json()  # Parse the JSON response
            else:
                print(f"Failed to fetch state, status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching state: {e}")
            return None

    def get_state(self):
        state_data = self.fetch_state()  # Get the latest state

        if state_data is None:
            return None  # If no data could be fetched, return None

        # Extract the job data and node data from the fetched state
        job_data = {
            'job_training_progress': state_data['job_training_progress'][0],  # Assume there is only one job
            'job_training_accuracy': state_data['job_training_accuracy'][0],  # Same as above
            'job_training_loss': state_data['job_training_loss'][0],  # Same as above
            'container_cpu_usage': state_data['container_used_cpu'][0],  # Same as above
            'container_mem_usage': state_data['container_used_mem'][0],  # Same as above
            'container_gpu_usage': state_data['container_used_gpu'][0],  # Same as above
        }

        # Extract node utilization data for CPU, Memory, and GPU
        node_data = {
            'node_cpu_utilization': state_data['node_cpu_utilization'],  # List of CPU utilizations per node
            'node_mem_utilization': state_data['node_mem_utilization'],  # List of memory utilizations per node
            'node_gpu_utilization': state_data['node_gpu_utilization'],  # List of GPU utilizations per node
        }

        # Combine the job and node data into one dictionary to return
        state = {**job_data, **node_data}
        return state

    def get_job_data(self):
        """Fetch the job-related data from the state."""
        state_data = self.fetch_state()  # Get the state
    
        if state_data is None:
            return None  # Return None if we couldn't fetch the state
    
        # Extract the job-related data from the state response
        return {
            'progress': state_data['job_training_progress'][0],  # Assuming there's only one job
            'accuracy': state_data['job_training_accuracy'][0],  # Same as above
            'loss': state_data['job_training_loss'][0],  # Same as above
            'container_cpu': state_data['container_used_cpu'][0],  # Same as above
            'container_mem': state_data['container_used_mem'][0],  # Same as above
            'container_gpu': state_data['container_used_gpu'][0],  # Same as above
        }

    def get_nodes(self):
        """Fetch the node-related data from the state."""
        state_data = self.fetch_state()  # Get the state
    
        if state_data is None:
            return None  # Return None if we couldn't fetch the state
    
        # Extract the node-related data from the state response
        nodes = []
        for i in range(len(state_data['node_cpu_utilization'])):
            nodes.append({
                'name': f'k8s-worker-{i+1}',
                'cpu': state_data['node_cpu_utilization'][i],
                'mem': state_data['node_mem_utilization'][i],
                'gpu': state_data['node_gpu_utilization'][i],
            })
        
        return nodes

    def step(self, action):
        """Apply the action (node selection) and calculate the new state and reward."""
        node_selected = action['node_name']
        node_score = action['node_score']
        # Simulate job scheduling by updating state (e.g., job allocation, progress).
        self.schedule_job_on_node(node_selected)  # This function would simulate job allocation.
        # Recalculate the state based on the node selected.
        self.state = self.get_state()
        # Calculate reward based on action taken (node selected).
        reward = self.calculate_reward()
        return self.state, reward, False  # Return new state, reward, and `done` (False for now)
    
    def schedule_job_on_node(self, node_selected):
        """Simulate the job scheduling on the selected node (update job progress, resources, etc.)."""
        # Implement job allocation logic here. For example:
        job_progress = self.job_data['progress'] + 10  # Increment job progress.
        if job_progress > 100:
            job_progress = 100  # Max job progress.
        self.job_data['progress'] = job_progress

    def calculate_reward(self):
        """Calculate reward based on job progress and node utilization."""
        job_progress = self.job_data['progress']
        node_utilization = self.calculate_node_utilization()
        reward = self.a1(job_progress) + self.a2(node_utilization)
        return reward
    
    def a1(self, job_progress):
        """Reward for job progress (faster job execution)."""
        return job_progress
    
    def a2(self, node_utilization):
        """Reward for optimal node utilization (maximizing resource efficiency)."""
        return 1 / (1 + node_utilization)  # Example formula

    def calculate_node_utilization(self):
        """Calculate overall node utilization based on available nodes."""
        total_utilization = 0
        for node in self.nodes:
            total_utilization += node['cpu'] + node['mem'] + node['gpu']
        return total_utilization / len(self.nodes)

