import os
import logging
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from deepLearningEnvironment import deepLearningEnvironment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

def log_rewards(step, reward, log_file='logs/training_rewards.log'):
    with open(log_file, 'a') as file:
        file.write(f"Step {step}, Reward: {reward}\n")
    logging.info(f"Step {step} - Reward logged: {reward}")

def train_and_track(env, total_steps=200_000, log_interval=10_000, eval_episodes=5, max_steps=100):
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_tensorboard")
    logging.info("Starting SAC training...")
    steps_done = 0

    while steps_done < total_steps:
        model.learn(total_timesteps=log_interval, reset_num_timesteps=False)
        steps_done += log_interval

        total_eval_reward = 0
        for ep in range(eval_episodes):
            obs = env.reset()
            ep_reward = 0
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                ep_reward += reward[0]  # reward is a vector, use [0] for scalar
                if done[0]:  # done is a vector of length 1
                    break
            total_eval_reward += ep_reward
            logging.info(f"Eval Episode {ep + 1}: Reward = {ep_reward}")

        avg_eval_reward = total_eval_reward / eval_episodes
        log_rewards(steps_done, avg_eval_reward)
        logging.info(f"Step {steps_done} - Avg Eval Reward: {avg_eval_reward}")

    model.save("sac_scheduler_model")
    logging.info("Training completed. Model saved as 'sac_scheduler_model'.")

if __name__ == '__main__':
    env = DummyVecEnv([lambda: Monitor(deepLearningEnvironment(), log_dir)])
    train_and_track(env)

