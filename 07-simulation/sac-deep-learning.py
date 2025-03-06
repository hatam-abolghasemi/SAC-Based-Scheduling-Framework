import gymnasium as gym
from stable_baselines3 import SAC
import os

from deepLearningEnvironment import deepLearningEnvironment

def log_rewards(episode, reward, log_file='training_rewards.log'):
    with open(log_file, 'a') as file:
        file.write(f"Episode {episode}, Reward: {reward}\n")

def train_and_track(env, total_timesteps=10000, log_interval=4):
    model = SAC("MlpPolicy", env, verbose=1)
    episode_rewards = []
    current_rewards = 0
    episode_count = 0

    obs, info = env.reset()
    for step in range(total_timesteps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        current_rewards += reward
        model.learn(total_timesteps=1, log_interval=0, reset_num_timesteps=False, progress_bar=False)

        if terminated or truncated:
            episode_rewards.append(current_rewards)
            log_rewards(episode_count + 1, current_rewards)
            current_rewards = 0
            episode_count += 1
            obs, info = env.reset()

    model.save("sac_scheduler_model")
    print("Training completed. Model saved as 'sac_scheduler_model'.")
    print(f"Total Episodes: {episode_count}")
    print("Rewards logged in 'training_rewards.log'.")

if __name__ == '__main__':
    env = deepLearningEnvironment()
    train_and_track(env)

