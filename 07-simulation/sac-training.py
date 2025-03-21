import gymnasium as gym
from stable_baselines3 import SAC
import os
import logging
from deepLearningEnvironment import deepLearningEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

def log_rewards(step, reward, log_file='logs/training_rewards.log'):
    with open(log_file, 'a') as file:
        file.write(f"Step {step}, Reward: {reward}\n")
    logging.info(f"Step {step} - Reward logged: {reward}")

def train_and_track(env, total_steps=10, max_steps=10):
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10)
    step_rewards = []
    logging.info("Starting training...")
    for step in range(total_steps):
        # logging.info(f"Step {step + 1} starting...")
        current_rewards = 0
        for episode in range(10):
            obs = env.reset()
            episode_rewards = 0
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                episode_rewards += reward
                model.learn(total_timesteps=1, reset_num_timesteps=False, progress_bar=False)
                if done:
                    logging.info(f"Episode {episode + 1} terminated at step {_ + 1}.")
                    break
            current_rewards += episode_rewards
            # logging.info(f"Episode {episode + 1} finished with reward: {episode_rewards}")
        step_rewards.append(current_rewards)
        log_rewards(step + 1, current_rewards)
        logging.info(f"Step {step + 1} completed with total reward: {current_rewards}")
    model.save("sac_scheduler_model")
    logging.info("Training completed. Model saved as 'sac_scheduler_model'.")
    logging.info(f"Total Steps: {total_steps}")
    logging.info(f"Rewards logged in 'training_rewards.log'.")


if __name__ == '__main__':
    env = deepLearningEnvironment()
    env = Monitor(env, "logs")
    env = DummyVecEnv([lambda: env])
    train_and_track(env)

