# state_fetcher.py
import requests
import numpy as np
from gym import spaces

def fetch_metrics():
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
        state = np.array(list(filtered_metrics.values()), dtype=np.float32)

        # Dynamically adjust observation space
        observation_space = spaces.Box(
            low=0.0, high=100.0, shape=state.shape, dtype=np.float32
        )

        return state, observation_space

    except Exception as e:
        print(f"Error fetching metrics: {e}")
        state = np.array([0.0, 0.0], dtype=np.float32)  # Default state
        observation_space = spaces.Box(
            low=0.0, high=100.0, shape=state.shape, dtype=np.float32
        )
        return state, observation_space

