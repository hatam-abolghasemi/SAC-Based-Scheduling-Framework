import logging
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from deepLearningEnvironment import deepLearningEnvironment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def perform_inference(env):
    model = SAC.load("sac_scheduler_model")
    model.set_env(env)

    obs = env.reset()
    total_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        total_reward += reward[0]  # reward is a list/array
        logging.info(f"Action taken: {action}, Reward: {reward[0]}, Total reward: {total_reward}")

        if done[0]:  # done is also a list/array
            logging.info(f"Episode finished with total reward: {total_reward}")
            obs = env.reset()
            total_reward = 0  # Reset reward counter for the new episode

if __name__ == '__main__':
    env = DummyVecEnv([lambda: deepLearningEnvironment()])
    perform_inference(env)

