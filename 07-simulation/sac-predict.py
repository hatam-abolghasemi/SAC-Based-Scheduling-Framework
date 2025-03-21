import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from deepLearningEnvironment import deepLearningEnvironment
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def perform_inference(env):
    # Load the trained model
    model = SAC.load("sac_scheduler_model")
    
    # Set the model's environment
    model.set_env(env)
    
    # Perform inference (in an infinite loop)
    obs = env.reset()  # Reset environment to get the initial observation
    total_reward = 0

    while True:  # Infinite loop for continuous inference
        action, _states = model.predict(obs, deterministic=False)  # Predict action based on the state
        obs, reward, done, info = env.step(action)  # Take action in the environment
        total_reward += reward  # Accumulate reward
        
        # Optionally print details of the action, reward, etc.
        logging.info(f"Action taken: {action}, Reward: {reward}, Total reward: {total_reward}")

        # Reset the environment when done, to keep it running indefinitely
        if done:
            logging.info(f"Episode finished with total reward: {total_reward}")
            obs = env.reset()  # Reset environment for the next "episode"

if __name__ == '__main__':
    # Create the environment
    env = deepLearningEnvironment()
    env = DummyVecEnv([lambda: env])

    # Perform infinite inference
    perform_inference(env)

