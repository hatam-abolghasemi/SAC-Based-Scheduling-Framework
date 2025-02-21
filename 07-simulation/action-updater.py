import time
import subprocess
import json
import random
from flask import Flask, jsonify

# Flask app initialization
app = Flask(__name__)

# Variable to store the best node name
best_node_name = ""

# Function to get the state from the API (via curl and grep)
def get_state():
    try:
        result = subprocess.run(
            ["curl", "http://0.0.0.0:9904/metrics"],
            capture_output=True, text=True, check=True
        )
        metrics = result.stdout.splitlines()
        utilization_data = {
            "node_cpu_utilization": [],
            "node_gpu_utilization": [],
            "node_mem_utilization": []
        }

        for line in metrics:
            if "node_cpu_utilization" in line:
                utilization_data["node_cpu_utilization"].append(float(line.split(" ")[1]))
            elif "node_gpu_utilization" in line:
                utilization_data["node_gpu_utilization"].append(float(line.split(" ")[1]))
            elif "node_mem_utilization" in line:
                utilization_data["node_mem_utilization"].append(float(line.split(" ")[1]))

        return utilization_data

    except subprocess.CalledProcessError as e:
        print(f"Error fetching metrics: {e}")
        return {}

# Function to score the nodes based on resource utilization
def score_nodes(state):
    node_scores = []
    
    cpu_util = state['node_cpu_utilization']
    gpu_util = state['node_gpu_utilization']
    mem_util = state['node_mem_utilization']
    
    for i in range(len(cpu_util)):
        cpu_score = cpu_util[i]
        gpu_score = gpu_util[i]  # GPU is weighted twice
        mem_score = mem_util[i]
        
        total_score = cpu_score + gpu_score * 2 + mem_score  # GPU gets double weight
        node_scores.append(total_score)

    min_score = min(node_scores)
    best_nodes = [i for i, score in enumerate(node_scores) if score == min_score]
    
    best_node_index = random.choice(best_nodes)
    
    return best_node_index, node_scores

# Function to update the top node (called every 3 seconds)
def update_top_node():
    global best_node_name  # Use the global variable

    while True:
        state = get_state()

        if not state:
            print("Error: Could not fetch state.")
            break

        best_node_index, node_scores = score_nodes(state)

        # Get the node name based on the index (e.g., k8s-worker-1)
        new_best_node_name = f"k8s-worker-{best_node_index + 1}"

        if new_best_node_name != best_node_name:
            best_node_name = new_best_node_name
            # print(f"Best node updated: {best_node_name}")

        time.sleep(3)

# Flask route to get the best node
@app.route('/best-node', methods=['GET'])
def get_best_node():
    return best_node_name

# Main function to run the script
def main():
    print("Action Updater started...")

    # Start the background task to update the best node
    import threading
    threading.Thread(target=update_top_node, daemon=True).start()

    # Run the Flask web server to serve the best node
    app.run(host="0.0.0.0", port=9908)

if __name__ == "__main__":
    main()

