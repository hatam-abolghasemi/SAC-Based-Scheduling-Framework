job-generator --> cluster --> scheduler --> node-exporter --> job-exporter-manager --> state-updater

Episode: A time period (e.g., 15 seconds) during which jobs are scheduled, their actions are taken, and rewards are calculated. The episode ends when the time period is over or when a set condition is met (like all jobs in the queue are processed or a specific number of jobs are scheduled).

Step: A single decision point within an episode where the system selects a node for a job, fetches the state, takes action, and calculates the reward. Each step represents one scheduling decision made for a job.

k8s-worker:

GPU Cores: 5120
Memory: 96 GiB
CPU: 32


job_id:             9
batch_size:         64
learning_rate:      0.001
requested_cpu:      4
requested_memory:   16Gi
requested_gpu:      1
dataset:            Cityscapes
framework:          TensorFlow
model:              SegNet

Average Memory Usage Breakdown (GiB) (16 GiB node total)
Epochs  Data Loading (GiB)  Forward Pass (GiB)  Backward Pass (GiB) Checkpoint Saving (GiB)
10      4.5 GiB             5.0 GiB             5.5 GiB             1.0 GiB
50      5.0 GiB             5.1 GiB             5.6 GiB             1.2 GiB
100     5.2 GiB             5.2 GiB             5.8 GiB             1.3 GiB
300     5.5 GiB             5.4 GiB             6.0 GiB             1.5 GiB
500     5.7 GiB             5.5 GiB             6.2 GiB             1.6 GiB

Average CPU Core Usage Breakdown (4 CPUs) (No change, included for completeness)
Epochs  Data Loading (Cores)    Forward Pass (Cores)    Backward Pass (Cores)   Checkpoint Saving (Cores)
10      2.5 cores               0.5 cores               0.7 cores               0.3 cores
50      2.7 cores               0.5 cores               0.8 cores               0.4 cores
100     2.8 cores               0.5 cores               0.9 cores               0.4 cores
300     2.9 cores               0.6 cores               1.0 cores               0.5 cores
500     3.0 cores               0.7 cores               1.1 cores               0.5 cores

Average GPU Core Usage Breakdown (2,560 CUDA Cores total)
Epochs  Data Loading (GPU Cores)	Forward Pass (GPU Cores)    Backward Pass (GPU Cores)   Checkpoint Saving (GPU Cores)
10      ~200 cores	                ~1800 cores                 ~2200 cores                 ~100 cores
50      ~220 cores	                ~1820 cores                 ~2230 cores                 ~120 cores
100     ~230 cores	                ~1850 cores                 ~2250 cores                 ~150 cores
300     ~250 cores	                ~1880 cores                 ~2280 cores                 ~170 cores
500     ~270 cores	                ~1900 cores                 ~2300 cores                 ~180 cores

Training Loss Breakdown (Cross-Entropy Loss)
Epochs  20% Progress (of Epochs)    40% Progress    60% Progress    80% Progress    100% Progress
10      1.85                        1.85            1.85            1.85            1.85
50      1.40                        1.35            1.30            1.20            1.30
100     1.10                        1.05            1.00            0.90            1.05
300     0.85                        0.80            0.75            0.70            0.72
500     0.75                        0.72            0.70            0.65            0.72

Training Accuracy Breakdown
Epochs  20% Progress (of Epochs)    40% Progress    60% Progress    80% Progress    100% Progress
10      65.0                        65.5            66.0            66.5            65.0
50      75.0                        75.5            77.0            78.0            75.5
100     80.5                        81.0            81.5            82.0            82.0
300     85.5                        86.5            87.0            88.5            89.5
500     89.0                        89.5            90.0            91.0            92.0

