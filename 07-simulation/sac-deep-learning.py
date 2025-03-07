import gymnasium as gym
from stable_baselines3 import SAC
import os
import logging
from deepLearningEnvironment import deepLearningEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def log_rewards(episode, reward, log_file='training_rewards.log'):
    with open(log_file, 'a') as file:
        file.write(f"Episode {episode}, Reward: {reward}\n")
    logging.info(f"Episode {episode} - Reward logged: {reward}")

def train_and_track(env, total_episodes=100, max_steps=200):
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    episode_rewards = []
    logging.info("Starting training...")
    for episode in range(total_episodes):
        logging.info(f"Episode {episode + 1} starting...")
        obs, info = env.reset()
        current_rewards = 0
        for step in range(max_steps):
            logging.info(f"Step {step + 1} for Episode {episode + 1}...")
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            current_rewards += reward
            model.learn(total_timesteps=1, log_interval=0, reset_num_timesteps=False, progress_bar=False)
            if terminated or truncated:
                logging.info(f"Episode {episode + 1} terminated at Step {step + 1}.")
                break
        episode_rewards.append(current_rewards)
        log_rewards(episode + 1, current_rewards)
        logging.info(f"Episode {episode + 1} completed with total reward: {current_rewards}")
    model.save("sac_scheduler_model")
    logging.info("Training completed. Model saved as 'sac_scheduler_model'.")
    logging.info(f"Total Episodes: {total_episodes}")
    logging.info("Rewards logged in 'training_rewards.log'.")

if __name__ == '__main__':
    env = deepLearningEnvironment()
    env = Monitor(env)  # Monitor wrapper
    env = DummyVecEnv([lambda: env])
    train_and_track(env)

