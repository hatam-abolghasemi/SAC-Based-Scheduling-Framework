import time
import requests

class JobExporter:
    def __init__(self, epochs, node_resources):
        self.epochs = epochs
        self.initial_resources = node_resources.copy()
        self.node_resources = node_resources
        self.passed_epochs = 0
        self.timer = 0

    @staticmethod
    def fetch_jobs():
        try:
            # Simulate a GET request to fetch jobs from an endpoint
            print("Fetching jobs from 0.0.0.0:9901/jobs...")  # Debugging line
            response = requests.get("http://0.0.0.0:9901/jobs")
            print(f"API response status: {response.status_code}")  # Debugging line
            
            if response.status_code == 200:
                jobs = response.json()
                # Filter jobs where job_id is 9
                return [job for job in jobs if job['job_id'] == 9]
            else:
                print(f"Error fetching jobs, status code: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error in fetching jobs: {e}")
            return []

    def get_stage_requirements(self):
        if self.passed_epochs < 10:
            mem = [4.5, 5.0, 5.5, 1.0]
            cpu = [2.5, 0.5, 0.7, 0.3]
            gpu = [200, 1800, 2200, 100]
        elif self.passed_epochs < 50:
            mem = [5.0, 5.1, 5.6, 1.2]
            cpu = [2.7, 0.5, 0.8, 0.4]
            gpu = [220, 1820, 2230, 120]
        elif self.passed_epochs < 100:
            mem = [5.2, 5.2, 5.8, 1.3]
            cpu = [2.8, 0.5, 0.9, 0.4]
            gpu = [230, 1850, 2250, 150]
        elif self.passed_epochs < 300:
            mem = [5.5, 5.4, 6.0, 1.5]
            cpu = [2.9, 0.6, 1.0, 0.5]
            gpu = [250, 1880, 2280, 170]
        else:
            mem = [5.7, 5.5, 6.2, 1.6]
            cpu = [3.0, 0.7, 1.1, 0.5]
            gpu = [270, 1900, 2300, 180]

        return [{'cpu': cpu[i], 'memory': mem[i], 'gpu': gpu[i]} for i in range(4)]

    def run_stage(self, stage_requirements):
        time.sleep(1)
        self.timer += 1
        for res, req in stage_requirements.items():
            if self.node_resources[res] < req:
                print(f"Warning: Resource allocation issue. Required {res}: {req}, available: {self.node_resources[res]}")
            self.node_resources[res] -= req

    def release_resources(self, stage_requirements):
        for res, req in stage_requirements.items():
            self.node_resources[res] += req

    def get_training_loss(self):
        epoch_thresholds = {
            10: 1.85,
            50: 1.30,
            100: 1.05,
            300: 0.72,
            500: 0.72,
        }

        for threshold, loss in epoch_thresholds.items():
            if self.passed_epochs <= threshold:
                previous_threshold = list(epoch_thresholds.keys())[
                    list(epoch_thresholds.keys()).index(threshold) - 1
                ] if threshold > 10 else 0
                
                progress = (self.passed_epochs - previous_threshold) / (threshold - previous_threshold)
                if progress < 0.2:
                    return loss
                elif progress < 0.4:
                    return loss - 0.05
                elif progress < 0.6:
                    return loss - 0.10
                elif progress < 0.8:
                    return loss - 0.15
                else:
                    return loss - 0.20

    def get_training_accuracy(self):
        epoch_thresholds = {
            10: 65.0,
            50: 75.0,
            100: 80.5,
            300: 85.5,
            500: 89.0,
        }

        for threshold, accuracy in epoch_thresholds.items():
            if self.passed_epochs <= threshold:
                previous_threshold = list(epoch_thresholds.keys())[
                    list(epoch_thresholds.keys()).index(threshold) - 1
                ] if threshold > 10 else 0
                
                progress = (self.passed_epochs - previous_threshold) / (threshold - previous_threshold)
                if progress < 0.2:
                    return accuracy
                elif progress < 0.4:
                    return accuracy + 0.5
                elif progress < 0.6:
                    return accuracy + 1.0
                elif progress < 0.8:
                    return accuracy + 1.5
                else:
                    return accuracy + 2.0

    def run(self):
        # Main loop to keep checking for jobs and process them
        while True:
            print("Checking for jobs with job_id == 9...")  # Debugging line
            jobs = self.fetch_jobs()
            if not jobs:
                print("No jobs found with job_id == 9. Retrying in 5 seconds...")  # Debugging line
                time.sleep(5)  # Retry after 5 seconds
                continue

            for job in jobs:
                generation_id = job['generation_id']
                print(f"Processing job with job_id: {job['job_id']} and generation_id: {generation_id}")
                self.epochs = job['required_epoch']
                self.node_resources = job.get('node_resources', self.node_resources)  # Ensure node_resources are available in the job
                self.passed_epochs = 0
                self.timer = 0

                while self.passed_epochs < self.epochs:
                    stage_requirements = self.get_stage_requirements()
                    for i, stage_name in enumerate(['data_loading', 'forward_pass', 'backward_pass', 'checkpoint_saving']):
                        self.run_stage(stage_requirements[i])
                        self.release_resources(stage_requirements[i])
                    
                    self.passed_epochs += 1
                    loss = self.get_training_loss()
                    accuracy = self.get_training_accuracy()
                    print(f"Epoch {self.passed_epochs}/{self.epochs} completed. Training Loss: {loss:.2f}, Training Accuracy: {accuracy:.2f}%")
                print(f"Job {job['job_id']} (Generation {generation_id}) completed in {self.timer} seconds.")

# Sample usage (this part can be outside the class or in the main entry point of the program)
# You can test by manually specifying a job or by simulating the API request for jobs.
initial_resources = {'cpu': 32, 'memory': 96, 'gpu': 5120}
job_exporter = JobExporter(epochs=140, node_resources=initial_resources)
job_exporter.run()

