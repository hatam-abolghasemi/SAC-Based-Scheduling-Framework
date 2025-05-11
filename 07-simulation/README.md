Start: root mode and source venv/bin/activate
nohup python3 job-generator.py > job-generator.log 2>&1 & nohup python3 cluster.py > cluster.log 2>&1 & nohup python3 node-exporter.py > node-exporter.log 2>&1 & nohup python3 job-exporter-manager-docker.py > job-exporter-manager-docker.log 2>&1 & nohup python3 state-updater.py > state-updater.log 2>&1 & nohup python3 sac-training.py > sac-training.log 2>&1 & nohup python3 prometheus-reloader.py > prometheus-reloader.log 2>&1 &

Stop:
pkill -f python3


job-generator --> cluster --> (optional: scheduler) --> node-exporter --> job-exporter-manager --> state-updater --> sac-training --> prometheus-reloader

k8s-worker:

GPU Cores: 5120
Memory: 96 GiB
CPU: 32

