1. Workload Definition and Monitoring
Input Source: Ensure the config.ini file is placed alongside the app source code, defining:
The dataset
The deep learning model
The deep learning framework
Workload Checker: Your existing workload_checker app reads and verifies these values from config.ini.
Prometheus Integration: Set up Prometheus queries to scrape real-time resource utilization data for each pod/workload. The data should include CPU, memory, and GPU usage labeled by the workload properties (dataset, model, framework).
2. SAC Algorithm for Optimization
State: Define the state as the resource utilization metrics gathered by Prometheus, such as CPU, memory, and GPU usage. Include:
Current workload resource utilization
Model-specific performance metrics (like training speed, validation loss, etc.)
Action: Actions are adjustments to the configuration parameters in config.ini, such as:
Batch size
Learning rate
Number of layers (if applicable)
Number of epochs or other tunable hyperparameters
Reward: Define the reward based on the increase in resource utilization efficiency. Higher utilization while maintaining or improving model performance (like lower validation loss) results in a higher reward.
3. Data Flow
Initial Data Collection: The JPO starts by reading the workload's config.ini to identify dataset, model, and framework.
Prometheus Queries: Use Prometheus to collect real-time timeseries data on resource utilization, keyed by workload properties. This becomes the input dataset for the SAC algorithm.
Historical Data: Optionally use historical resource data for better decision-making, particularly for workloads that have run before.
4. JPO Training & Prediction Loop
Training the SAC Model:
Episode: Each workload execution can be treated as an episode. The JPO gathers resource utilization data during the run and suggests updates based on how efficiently the resources were used.
Continuous Learning: Keep the SAC agent in continuous learning mode, adjusting to the dynamic nature of resource availability in a Kubernetes cluster.
Inference (Action Suggestion):
Once the SAC model has been trained or partially trained, use it to suggest new hyperparameters for the upcoming workload.
These suggestions are automatically written to the config.ini.
5. CI/CD Integration
CI/CD Stage 1 - JPO: The JPO runs as the first step in the CI/CD pipeline.
Load config.ini: The JPO reads the current config.ini and Prometheus resource data.
Optimization: The SAC algorithm suggests new values for hyperparameters such as batch size and learning rate.
Update config.ini: The updated config.ini is saved with the optimized parameters.
CI/CD Stage 2 - Build: The pipeline proceeds to the build stage with the optimized parameters.
6. Parameter Constraints
Limits and Ranges: Implement a constraints module to ensure the new parameter values are within acceptable ranges (e.g., batch size ≥ 1, learning rate between 0.0001 and 0.1).
Validation: Validate the config.ini after JPO suggestions to prevent invalid configurations from entering the next CI/CD stage.
7. Testing & Evaluation
Simulated Workloads: Initially, test the JPO on simulated Kubernetes workloads with varying resource demands to evaluate how well it adjusts parameters based on resource usage.
Performance Metrics: Evaluate performance improvements by comparing before-and-after resource utilization and model performance metrics (like training time and accuracy).
8. Logging & Feedback
Logging: Ensure that every suggested change to the config.ini is logged, along with the corresponding resource usage and performance data, to enable monitoring of JPO effectiveness.
Feedback Loop: If the JPO’s suggestions result in degraded model performance (e.g., worse validation loss), feed this information back into the reward system, penalizing such outcomes.
