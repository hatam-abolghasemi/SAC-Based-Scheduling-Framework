# Metrics Fetcher

This Flask application fetches CPU and memory utilization metrics from a Prometheus server and exposes them via a /metrics endpoint. The metrics include:

* Node CPU Utilization
* Node Memory Utilization
* Container CPU Utilization

## Requirements
* Python 3.6 or higher
* Flask
* Requests

You can install the required packages using pip:
```bash
pip install Flask requests
```

## Configuration
* Before running the application, ensure that you configure the following variables in the code:
* `PROMETHEUS_URL`: The URL of your Prometheus server. By default, it is set to http://0.0.0.0:9090.
* Update the Prometheus queries (`NODE_CPU_UTILIZATION_QUERY`, `NODE_MEMORY_UTILIZATION_QUERY`, `CONTAINER_CPU_UTILIZATION_QUERY`) as needed.

```bash
python app.py
```

* The application will start running on http://0.0.0.0:4223.

Access the metrics at http://<your-server-ip>:4223/metrics.

## How It Works
* The application runs a background thread that fetches metrics from Prometheus every 15 seconds.
* It retrieves:
    * Node CPU Utilization: Average CPU utilization percentage across all nodes.
    * Node Memory Utilization: Average memory utilization percentage across all nodes.
    * Container CPU Utilization: CPU utilization percentage for each container, labeled by dataset, framework, and model.
* The /metrics endpoint returns data in the following format:
```
node_cpu_utilization
node_memory_utilization
container_cpu_utilization(image:<image_name>,container:<name>,dataset:<dataset>,framework:<framework>,model:<model>)
```

