import requests
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

class ReScheduleEnv(Env):
    def __init__(self):
        # Action space: 0 = re-schedule False, 1 = re-schedule True
        self.action_space = Discrete(2)
        # Observation space: we’ll monitor CPU and memory utilization
        self.observation_space = Box(low=np.array([0, 0], dtype=np.float32), high=np.array([100, 100], dtype=np.float32))
        self.state = self.get_current_metrics()
        self.done = False

    def get_current_metrics(self):
        """Fetch metrics from the endpoint and return CPU and memory utilization as a state."""
        try:
            response = requests.get("http://127.0.0.1:4223/metrics")
            response.raise_for_status()
            metrics = response.text.splitlines()
            
            # Default values if metrics not found
            cpu_utilization = 50.0
            memory_utilization = 50.0
            
            # Extract CPU and memory utilization
            for metric in metrics:
                if "node_cpu_utilization" in metric:
                    cpu_utilization = float(metric.split()[-1].replace('%', ''))
                elif "node_memory_utilization" in metric:
                    memory_utilization = float(metric.split()[-1].replace('%', ''))
            
            return np.array([cpu_utilization, memory_utilization], dtype=np.float32)
        except requests.RequestException as e:
            print("Error fetching metrics:", e)
            # Return nominal values if metrics cannot be fetched
            return np.array([50.0, 50.0], dtype=np.float32)
    
    def step(self, action):
        # Update state by fetching current metrics
        self.state = self.get_current_metrics()
        
        # Check if CPU and memory utilization are within the target range (80 to 98)
        in_range = np.all((self.state >= 80) & (self.state <= 98))
        
        # Define reward based on whether metrics are in the target range
        reward = 1 if in_range else -1

        # No specific episode length or end condition in this scenario; define as needed
        self.done = False  # Can be set to True if you want to end the episode based on a condition
        info = {"action_taken": "re-schedule" if action == 1 else "no re-schedule"}
        
        return self.state, reward, self.done, info

    def reset(self):
        # Fetch initial state
        self.state = self.get_current_metrics()
        self.done = False
        return self.state

env = ReScheduleEnv()

# Simulate environment interaction
episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()  # Randomly choose to re-schedule or not
        n_state, reward, done, info = env.step(action)
        score += reward
        print(f"State: {n_state}, Action: {'Re-schedule' if action == 1 else 'No re-schedule'}, Reward: {reward}, Info: {info}")
    
    print(f'Episode:{episode} Score:{score}')

