import gymnasium as gym
from gymnasium import spaces
import numpy as np

class templateEnvironment(gym.Env):
    def __init__(self):
        super(templateEnvironment, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.state = np.array([0.0], dtype=np.float32)
        self.target_position = 5.0

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(-10, 10, size=(1,))
        return self.state, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state += action
        reward = -abs(self.state - self.target_position)
        done = False
        if self.state < -10 or self.state > 10:
            done = True
        return self.state, reward, done, False, {}

    def render(self):
        print(f"Current position: {self.state[0]}")

