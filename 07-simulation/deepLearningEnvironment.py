import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests

class deepLearningEnvironment(gym.Env):
    def __init__(self, max_dim=100):
        super(deepLearningEnvironment, self).__init__()
        self.max_dim = max_dim
        self.current_dim = 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(max_dim,), dtype=np.float32)
        self.state = np.zeros((self.current_dim,), dtype=np.float32)
        self.target_position = np.zeros((self.max_dim,), dtype=np.float32)

    def fetch_state(self):
        response = requests.get("http://0.0.0.0:9907/state")
        return np.array(response.json(), dtype=np.float32)

    def set_state_dimension(self):
        self.state = self.fetch_state()
        self.current_dim = len(self.state)
        self.target_position = np.zeros((self.current_dim,), dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        self.set_state_dimension()
        padded_state = np.zeros((self.max_dim,), dtype=np.float32)
        padded_state[:self.current_dim] = self.state
        return padded_state, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if action.shape != self.state.shape:
            action = np.resize(action, self.state.shape)
        self.state += action
        self.target_position = np.zeros((self.current_dim,), dtype=np.float32)
        reward = -np.sum(np.abs(self.state - self.target_position))
        done = np.any(self.state < -10) or np.any(self.state > 10)
        padded_state = np.zeros((self.max_dim,), dtype=np.float32)
        padded_state[:self.current_dim] = self.state
        return padded_state, reward, done, False, {}

    def render(self):
        print(f"Current state (dim {self.current_dim}): {self.state}")

