import gymnasium as gym
import numpy as np
import requests
import json
import time
from gymnasium import spaces
import random
import logging

job_generator_url = 'http://0.0.0.0:9902/jobs'
job_deploy_url = 'http://0.0.0.0:9901/deploy_job'
queue_url = 'http://0.0.0.0:9902/queue'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class deepLearningEnvironment(gym.Env):
    def __init__(self):
        super(deepLearningEnvironment, self).__init__()
        self.state = np.zeros((1,), dtype=np.float32)
        self.target_position = np.zeros((1,), dtype=np.float32)
        self.scheduled_generation_ids = set()
        self.generated_jobs = []
        self.current_step = 0
        self.max_steps = 200
        self.episode_start_time = time.time()
        self.state_dimension = None
        self.set_state_dimension()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.current_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.current_dim,), dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        self.scheduled_generation_ids.clear()
        self.state = self.fetch_state()
        self.current_dim = len(self.state)
        self.current_step = 0
        padded_state = np.zeros((self.current_dim,), dtype=np.float32)
        padded_state[:self.current_dim] = self.state[:self.current_dim]
        self.episode_start_time = time.time()
        logging.info("Step reset. Initial state fetched.")
        return padded_state, {}

    def step(self, action):
        max_retries = 5
        base_delay = 1.0
        self.state = self.fetch_state()
        for attempt in range(max_retries):
            self.current_dim = len(self.state)
            if self.current_dim == self.observation_space.shape[0]:
                break
            else:
                print(f"[Retry {attempt + 1}/{max_retries}] State size mismatch detected. "
                      f"Expected {self.observation_space.shape[0]}, got {self.current_dim}. Updating observation space.")
                self.update_observation_space(self.state)
                time.sleep(base_delay * (2 ** attempt))
        if self.current_dim != self.observation_space.shape[0]:
            print("Max retries reached. Proceeding with mismatched dimensions.")
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.current_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.current_dim,), dtype=np.float32)
        logging.debug(f"Step {self.current_step}: Received action: {action}")
        logging.debug("Fetching current state...")
        logging.debug(f"State before scheduling: {self.state}")
        logging.debug(f"Action applied: {action}")
        self.schedule_jobs(action)
        action = action[:self.state_dimension]
        self.state += action
        reward = self.reward_function()
        logging.debug(f"Reward calculated: {reward}")
        time.sleep(1)
        padded_state = np.zeros((self.current_dim,), dtype=np.float32)
        padded_state[:self.current_dim] = self.state[:self.current_dim]
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        if terminated:
            duration = time.time() - self.episode_start_time
            logging.info(f"Step finished after {self.current_step} steps. Duration: {duration:.2f} seconds.")
        return padded_state, reward, terminated, truncated, {}
    
    def render(self):
        print(f"Step: {self.current_step}, State: {self.state}")

    def update_observation_space(self, state):
        obs_shape = (len(state),)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    def set_state_dimension(self):
        self.state_dimension = len(self.state)

    # STEP 1: STATE --------------------------------------------------------------------------------------------------------------------------------
    def fetch_state(self):
        response = requests.get("http://0.0.0.0:9907/state")
        return np.array(response.json(), dtype=np.float32)

    def set_state_dimension(self):
        self.state = self.fetch_state()
        self.current_dim = len(self.state)
        self.target_position = np.zeros((self.current_dim,), dtype=np.float32)

    # STEP 2: ACTION --------------------------------------------------------------------------------------------------------------------------------
    def fetch_generated_jobs(self):
        max_retries = 5
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = requests.get(job_generator_url)
                if response.status_code == 200:
                    raw_response = response.text
                    jobs = []
                    for line in raw_response.splitlines():
                        try:
                            job = json.loads(line.replace("'", '"'))
                            jobs.append(job)
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to decode line: {line}")
                            logging.error(f"Error: {e}")
                    return jobs
                else:
                    logging.warning(f"Failed to fetch jobs, status code: {response.status_code}")
            except requests.RequestException as e:
                logging.warning(f"Error accessing job generator service. Retrying in {base_delay * (2 ** attempt)} seconds...")
                time.sleep(base_delay * (2 ** attempt))
        logging.error("Failed to fetch jobs after multiple retries.")
        return []

    def score_nodes(self, state):
        node_scores = []
        num_nodes = 13
        for i in range(num_nodes):
            cpu_util = state[i]
            gpu_util = state[i + 13]
            mem_util = state[i + 26]
            score = cpu_util + gpu_util + mem_util
            node_scores.append((i + 1, score))
        min_score = min(node_scores, key=lambda x: x[1])[1]
        best_nodes = [node for node, score in node_scores if score == min_score]
        logging.info(f"Scored nodes: {node_scores}. Best nodes: {best_nodes}")
        return random.choice(best_nodes)

    def schedule_jobs(self, action):
        jobs = self.fetch_generated_jobs()
        state = self.fetch_state()
        for job in jobs:
            generation_id = job['generation_id']
            if generation_id in self.scheduled_generation_ids:
                continue
            best_node = self.score_nodes(state)
            schedule_moment = job.get('generation_moment')
            job_data = {
                'generation_id': generation_id,
                'job_id': job.get('job_id'),
                'node': f"k8s-worker-{best_node}",
                'required_epoch': job.get('required_epoch'),
                'generation_moment': job.get('generation_moment'),
                'schedule_moment': schedule_moment
            }
            while True:
                try:
                    deploy_response = requests.post(job_deploy_url, json=job_data)
                    if deploy_response.status_code == 200:
                        print(f"Job with generation_id {generation_id} scheduled on node k8s-worker-{best_node} successfully.")
                        self.scheduled_generation_ids.add(generation_id)
                        queue_response = requests.post(queue_url, json={'generation_id': generation_id})
                        if queue_response.status_code != 200:
                            print(f"Failed to notify job-generator for job {job.get('job_id')}.")
                        break
                    else:
                        print(f"Failed to deploy job with generation_id {generation_id}. Response: {deploy_response.text}")
                except requests.RequestException as e:
                    print(f"Error accessing cluster. Retrying in 1 seconds...")
                time.sleep(1)

    # STEP 3: REWARD --------------------------------------------------------------------------------------------------------------------------------
    def calculate_average(self, values, start_index, step, count):
        indices = [start_index + (i * step) for i in range(count) if start_index + (i * step) < len(values)]
        if indices:
            return sum(values[idx - 1] for idx in indices) / len(indices)
        return 0.0

    def reward_function(self):
        state = self.fetch_state()
        progress_reward = self.calculate_average(state, 45, 15, 101)
        accuracy_reward = self.calculate_average(state, 44, 15, 101)
        loss_penalty = self.calculate_average(state, 43, 15, 101)
        elapsed_time_penalty = self.calculate_average(state, 48, 15, 101)
        cpu_utilization_reward = self.calculate_average(state, 1, 15, 3)
        gpu_utilization_reward = self.calculate_average(state, 2, 15, 3)
        mem_utilization_reward = self.calculate_average(state, 3, 15, 3)
        reward = (progress_reward + accuracy_reward +
                   cpu_utilization_reward + gpu_utilization_reward + mem_utilization_reward -
                   loss_penalty - elapsed_time_penalty)
        return float(reward)

