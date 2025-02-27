import gymnasium as gym
from stable_baselines3 import SAC
from deepLearningEnvironment import deepLearningEnvironment  # Assuming the custom environment is saved in deepLearningEnvironment.py

env = deepLearningEnvironment()

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_1d_example")
del model
model = SAC.load("sac_1d_example")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
