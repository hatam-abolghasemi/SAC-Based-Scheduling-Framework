import requests
import json

def fetch_and_process_state():
    # Fetch the state dynamically from the endpoint
    response = requests.get('http://0.0.0.0:9907/state')

    if response.status_code == 200:
        try:
            # Replace single quotes with double quotes to make the response valid JSON
            response_text = response.text.replace("'", '"')
            state_data = json.loads(response_text)  # Use json.loads to parse the string

            # Flatten the state data into a 1D vector
            state_vector = (
                state_data.get('node_used_cpu', []) +
                state_data.get('node_used_gpu', []) +
                state_data.get('node_used_mem', []) +
                state_data.get('container_used_cpu', []) +
                state_data.get('container_used_gpu', []) +
                state_data.get('container_used_mem', []) +
                state_data.get('job_training_loss', []) +
                state_data.get('job_training_accuracy', [])
            )

            # Print the processed state to stdout
            print(state_vector)

        except ValueError as e:
            print(f"Error processing the state data: {e}")
    else:
        print(f"Failed to fetch state data, status code: {response.status_code}")

if __name__ == "__main__":
    fetch_and_process_state()

