import logging
import subprocess
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from deepLearningEnvironment import deepLearningEnvironment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def count_scheduled_jobs():
    try:
        result = subprocess.check_output(['ss', '-nlpt'], stderr=subprocess.DEVNULL)
        lines = result.decode().splitlines()
        ports = set()

        for line in lines:
            if 'LISTEN' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith(':::') or part.startswith('0.0.0.0:') or part.startswith('127.0.0.1:'):
                        try:
                            port = int(part.split(':')[-1])
                            if 11001 <= port <= 12000:
                                ports.add(port)
                        except ValueError:
                            continue
        return len(ports)
    except Exception as e:
        logging.warning(f"Error counting jobs via ss: {e}")
        return 0

def perform_inference(env, max_jobs=1000):
    model = SAC.load("sac_successful_train/sac_scheduler_model")
    model.set_env(env)

    obs = env.reset()
    total_reward = 0

    while True:
        current_jobs = count_scheduled_jobs()
        if current_jobs >= max_jobs:
            logging.info(f"Reached {current_jobs} scheduled jobs. Stopping.")
            break

        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        total_reward += reward[0]
        logging.info(f"Action taken: {action}, Reward: {reward[0]}, Total reward: {total_reward}, Jobs scheduled: {current_jobs}/{max_jobs}")

        if done[0]:
            logging.info(f"Episode finished with total reward: {total_reward}")
            obs = env.reset()
            total_reward = 0

if __name__ == '__main__':
    env = DummyVecEnv([lambda: deepLearningEnvironment()])
    perform_inference(env)

