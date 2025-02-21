from stable_baselines3 import SAC
from stable_baselines3.common.envs import DummyVecEnv
import numpy as np

env = SchedulingEnvironment()
env = DummyVecEnv([lambda: env])
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
state = env.reset()
done = False
while not done:
    action, _states = model.predict(state)
    state, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")

