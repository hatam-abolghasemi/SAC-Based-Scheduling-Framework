import time
import gym
from state_fetcher import fetch_metrics
from reward import calculate_reward  # Import the calculate_reward function

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Initialize observation and state variables
        self.state, self.observation_space = fetch_metrics()

        # Action space (continuous example)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)

        # Time step counter
        self.timestep = 0

    def step(self, action):
        """
        Apply the action to the environment and return the next state, reward, done, and info.
        """
        # Fetch new state
        self.state, _ = fetch_metrics()

        # Calculate reward based on the current state using calculate_reward
        reward = calculate_reward(self.state)  # Using the state from fetch_metrics

        # Increment timestep
        self.timestep += 1

        # Define done condition (infinite training means it's always False)
        done = False

        # Additional info (optional)
        info = {}

        return self.state, reward, done, info

    def reset(self):
        """Reset the environment to its initial state."""
        self.state, _ = fetch_metrics()
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
        print(f"Reward: {reward}")  # Print the reward for each step

