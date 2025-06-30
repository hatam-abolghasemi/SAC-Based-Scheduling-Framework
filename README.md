SAC Predict:
nohup python3 job-generator-pattern.py > job-generator-pattern.log 2>&1 & nohup python3 cluster.py > cluster.log 2>&1 & nohup python3 node-exporter.py > node-exporter.log 2>&1 & nohup python3 job-exporter-manager-docker.py > job-exporter-manager-docker.log 2>&1 & nohup python3 state-updater.py > state-updater.log 2>&1 & nohup python3 sac-predict.py > sac-predict.log 2>&1 & nohup python3 prometheus-reloader.py > prometheus-reloader.log 2>&1 & nohup python3 general-exporter.py > general-exporter.log 2>&1 &

