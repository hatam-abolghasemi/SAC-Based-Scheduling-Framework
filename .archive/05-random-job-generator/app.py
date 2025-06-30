import os
import random
import subprocess
import time
from threading import Thread

def apply_manifest(manifest_path):
    try:
        subprocess.run(["kubectl", "apply", "-f", manifest_path], check=True)
        print(f"Applied: {manifest_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error applying {manifest_path}: {e}")

def apply_random_manifests(manifests_dir):
    while True:
        # Get a list of all manifest files in the directory
        manifests = [os.path.join(manifests_dir, f) for f in os.listdir(manifests_dir) if f.endswith(".yaml") or f.endswith(".yml")]
        
        if not manifests:
            print("No manifests found in the directory. Exiting.")
            break

        # Randomly select 1 to 5 manifests
        selected_manifests = random.sample(manifests, k=random.randint(1, min(5, len(manifests))))

        # Apply manifests concurrently
        threads = []
        for manifest in selected_manifests:
            thread = Thread(target=apply_manifest, args=(manifest,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Wait for a random interval between 5 and 300 seconds
        wait_time = random.randint(5, 300)
        print(f"Waiting for {wait_time} seconds before the next apply.")
        time.sleep(wait_time)

if __name__ == "__main__":
    manifests_directory = os.path.join(os.path.dirname(__file__), "manifests")
    apply_random_manifests(manifests_directory)

