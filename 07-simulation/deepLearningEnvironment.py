import gymnasium as gym
import numpy as np
import requests
import json
import time
from gymnasium import spaces
import random

job_generator_url = 'http://0.0.0.0:9902/jobs'
job_deploy_url = 'http://0.0.0.0:9901/deploy_job'
queue_url = 'http://0.0.0.0:9902/queue'

class deepLearningEnvironment(gym.Env):
    # STEP 0: MAIN --------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, max_dim=100):
        super(deepLearningEnvironment, self).__init__()
        self.max_dim = max_dim
        self.current_dim = 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(max_dim,), dtype=np.float32)
        self.state = np.zeros((self.current_dim,), dtype=np.float32)
        self.target_position = np.zeros((self.max_dim,), dtype=np.float32)
        self.scheduled_generation_ids = set()
        self.generated_jobs = []

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        self.state = self.fetch_state()
        self.current_dim = len(self.state)
        padded_state = np.zeros((self.max_dim,), dtype=np.float32)
        padded_state[:self.current_dim] = self.state[:self.max_dim]
        return padded_state, {}

    def step(self, action):
        time.sleep(15)
        self.state = self.fetch_state()
        self.current_dim = len(self.state)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if action.shape != self.state.shape:
            action = np.resize(action, self.state.shape)
        self.schedule_jobs(action)
        self.state += action
        reward = -np.sum(np.abs(self.state - self.target_position[:self.current_dim]))
        done = False
        padded_state = np.zeros((self.max_dim,), dtype=np.float32)
        padded_state[:self.current_dim] = self.state[:self.max_dim]
        return padded_state, reward, done, False, {}
    
    def render(self):
        print(f"Current state (dim {self.current_dim}): {self.state}")
   
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
        while True:
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
                            print(f"Failed to decode line: {line}")
                            print(f"Error: {e}")
                    return jobs
                else:
                    print(f"Failed to fetch jobs, status code: {response.status_code}")
            except requests.RequestException as e:
                print(f"Error accessing job generator service. Retrying in 1 seconds...")
            time.sleep(1)

    def score_nodes(self, state):
        node_scores = []
        num_nodes = 39
        for i in range(num_nodes):
            cpu_util = state[i]
            gpu_util = state[i + 13]
            mem_util = state[i + 26]
            score = cpu_util + gpu_util + mem_util
            node_scores.append((i + 1, score))  # node indices are 1-based
        min_score = min(node_scores, key=lambda x: x[1])[1]
        best_nodes = [node for node, score in node_scores if score == min_score]
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
    # Reward is calculated within the step function: -np.sum(np.abs(self.state - self.target_position))

