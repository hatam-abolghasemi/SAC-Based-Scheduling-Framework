import gym
from gym import spaces
import numpy as np
import requests

class RLSchedulerEnv(gym.Env):
    def __init__(self, num_nodes, num_jobs):
        super(RLSchedulerEnv, self).__init__()

        # Environment parameters
        self.num_nodes = num_nodes
        self.num_jobs = num_jobs
        self.nodes = [{'cpu': 0.0, 'mem': 0.0} for _ in range(num_nodes)]
        self.jobs = [{'status': 0} for _ in range(num_jobs)]  # 0: unscheduled, 1: running
        self.node_assignments = [-1] * num_jobs  # -1: unassigned, else: node index

        # State space: Flattened representation of nodes' and jobs' statuses
        self.state_size = 2 * self.num_nodes + self.num_jobs
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32)

        # Action space: Two types of actions: rescheduling and scheduling
        # Actions: [reschedule(job_id, node_id), schedule(job_id, node_id)]
        self.action_space = spaces.MultiDiscrete([self.num_jobs, self.num_nodes])

        # Metrics Fetcher URL
        self.metric_fetcher_url = "http://0.0.0.0:4223/metrics"

    def reset(self):
        # Reset all jobs and nodes to initial state
        self.nodes = [{'cpu': 0.0, 'mem': 0.0} for _ in range(self.num_nodes)]
        self.jobs = [{'status': 0} for _ in range(self.num_jobs)]
        self.node_assignments = [-1] * self.num_jobs
        return self._get_state()

    def step(self, action):
        job_id, node_id = action
        reward = 0.0

        # Reschedule or schedule job
        if self.jobs[job_id]['status'] == 1:  # Reschedule
            old_node = self.node_assignments[job_id]
            if old_node != node_id:  # Only reschedule if the node is different
                self._remove_job_from_node(job_id, old_node)
                self._assign_job_to_node(job_id, node_id)
                reward = self._calculate_reward()
        elif self.jobs[job_id]['status'] == 0:  # Schedule
            self._assign_job_to_node(job_id, node_id)
            reward = self._calculate_reward()

        # End condition: All jobs scheduled
        done = all(job['status'] == 1 for job in self.jobs)

        return self._get_state(), reward, done, {}

    def _get_state(self):
        # Fetch metrics and update node states
        metrics = self._fetch_metrics()
        for i, node in enumerate(self.nodes):
            node['cpu'] = metrics.get(f'node_{i}_cpu', 0.0)
            node['mem'] = metrics.get(f'node_{i}_mem', 0.0)

        # Flatten the state
        state = []
        for node in self.nodes:
            state.extend([node['cpu'], node['mem']])
        state.extend(job['status'] for job in self.jobs)
        return np.array(state, dtype=np.float32)

    def _assign_job_to_node(self, job_id, node_id):
        self.jobs[job_id]['status'] = 1
        self.node_assignments[job_id] = node_id

    def _remove_job_from_node(self, job_id, node_id):
        self.jobs[job_id]['status'] = 0
        self.node_assignments[job_id] = -1

    def _calculate_reward(self):
        # Reward function: Penalize high CPU/memory utilization and encourage balanced loads
        reward = 0.0
        for node in self.nodes:
            cpu_penalty = max(0, node['cpu'] - 0.8)  # Penalize CPU > 80%
            mem_penalty = max(0, node['mem'] - 0.8)  # Penalize Memory > 80%
            reward -= (cpu_penalty + mem_penalty)
        return reward

    def _fetch_metrics(self):
        # Fetch metrics from the metric-fetcher
        try:
            response = requests.get(self.metric_fetcher_url)
            if response.status_code == 200:
                metrics = response.text.split("\n")
                parsed_metrics = {}
                for metric in metrics:
                    if metric.strip():  # Skip empty lines
                        parts = metric.split()
                        if len(parts) >= 3:
                            metric_name = parts[1]
                            metric_value = float(parts[2].replace('%', ''))  # Remove '%' if present
                            parsed_metrics[metric_name] = metric_value / 100  # Normalize 0-1
                return parsed_metrics
        except Exception as e:
            print(f"Error fetching metrics: {e}")
        return {}

    def render(self, mode="human"):
        print(f"Nodes: {self.nodes}")
        print(f"Jobs: {self.jobs}")
        print(f"Node Assignments: {self.node_assignments}")

# Initializing and Random Testing

env = RLSchedulerEnv(num_nodes=3, num_jobs=1)
state = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action for testing
    state, reward, done, info = env.step(action)
    env.render()
    print(f"Action: {action}, Reward: {reward}")

