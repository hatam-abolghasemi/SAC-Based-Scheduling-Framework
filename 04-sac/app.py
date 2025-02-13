import gym
from gym import spaces
import numpy as np
import time
import requests

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Observation space dynamically adjusted based on metrics
        self.observation_space = None

        # Action space (continuous example)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # State initialization
        self.state = None

        # Time step counter
        self.timestep = 0

    def fetch_metrics(self):
        """Fetch metrics from the given endpoint and parse the state."""
        try:
            response = requests.get("http://0.0.0.0:4223/metrics")
            response.raise_for_status()

            # Parse the metrics into a dictionary
            metrics = {}
            for line in response.text.strip().split("\n"):
                parts = line.split()
                if len(parts) == 3:
                    _, key, value = parts
                    value = float(value.strip('%'))  # Remove % and convert to float

                    # Validate and normalize metric values to [0, 100]
                    if 0.0 <= value <= 100.0:
                        metrics[key] = round(value, 2)  # Keep two decimal places

            # Filter for specific metrics (node_cpu_utilization, node_memory_utilization)
            filtered_metrics = {
                "node_cpu_utilization": metrics.get("node_cpu_utilization", 0.0),
                "node_memory_utilization": metrics.get("node_memory_utilization", 0.0)
            }

            # Convert metrics to state
            self.state = np.array(list(filtered_metrics.values()), dtype=np.float32)

            # Dynamically adjust observation space
            self.observation_space = spaces.Box(
                low=0.0, high=100.0, shape=self.state.shape, dtype=np.float32
            )

        except Exception as e:
            print(f"Error fetching metrics: {e}")
            self.state = np.array([0.0, 0.0], dtype=np.float32)  # Default state

    def step(self, action):
        """
        Apply the action to the environment and return the next state, reward, done, and info.
        """
        # Fetch new state
        self.fetch_metrics()

        # Placeholder: Calculate reward (to be defined later)
        reward = 0  # Example: Placeholder reward

        # Increment timestep
        self.timestep += 1

        # Define done condition (infinite training means it's always False)
        done = False

        # Additional info (optional)
        info = {}

        return self.state, reward, done, info

    def reset(self):
        """Reset the environment to its initial state."""
        self.fetch_metrics()
        self.timestep = 0
        return self.state

    def render(self, mode='human'):
        """Render the environment (optional)."""
        print(f"Timestep: {self.timestep}, State: {self.state}")

    def close(self):
        """Clean up resources (optional)."""
        pass

# Test the environment
if __name__ == "__main__":
    env = CustomEnv()
    state = env.reset()
    while True:  # Simulate continuous operation
        print("Fetching new state...")
        time.sleep(15)  # Wait for 15 seconds
        action = env.action_space.sample()  # Sample a random action
        state, reward, done, info = env.step(action)
        env.render()

