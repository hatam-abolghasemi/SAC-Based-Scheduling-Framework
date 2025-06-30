import os
import configparser
import sys

# Set the config.ini and Dockerfile names
CONFIG_FILE = "config.ini"
DOCKERFILE = "Dockerfile"

def extract_workload_from_config(config_file):
    """Extract the framework, dataset, and model from the [WORKLOAD] section in config.ini."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return (
        config.get('WORKLOAD', 'framework', fallback=None),
        config.get('WORKLOAD', 'dataset', fallback=None),
        config.get('WORKLOAD', 'model', fallback=None),
        config.getint('TRAINING', 'batch_size', fallback=None),
        config.getfloat('TRAINING', 'learning_rate', fallback=None),
        config.getint('TRAINING', 'num_epochs', fallback=None)
    )

def find_config_file(directory):
    """Find config.ini in the specified directory."""
    config_path = os.path.join(directory, CONFIG_FILE)
    return config_path if os.path.exists(config_path) else None

def find_dockerfile(directory):
    """Find Dockerfile in the specified directory."""
    dockerfile_path = os.path.join(directory, DOCKERFILE)
    return dockerfile_path if os.path.exists(dockerfile_path) else None

def check_and_update_dockerfile(dockerfile_path, framework, dataset, model, batch_size, learning_rate, num_epochs):
    """Check if Dockerfile contains framework, dataset, model, batch_size, learning_rate, and num_epochs labels, and update if missing."""
    with open(dockerfile_path, 'r') as file:
        dockerfile_lines = file.readlines()

    # Check if LABELs for framework, dataset, model, batch_size, learning_rate, and num_epochs exist
    has_framework_label = any('LABEL framework=' in line for line in dockerfile_lines)
    has_dataset_label = any('LABEL dataset=' in line for line in dockerfile_lines)
    has_model_label = any('LABEL model=' in line for line in dockerfile_lines)
    has_batch_size_label = any('LABEL batch_size=' in line for line in dockerfile_lines)
    has_learning_rate_label = any('LABEL learning_rate=' in line for line in dockerfile_lines)
    has_num_epochs_label = any('LABEL num_epochs=' in line for line in dockerfile_lines)

    # Append missing labels before the last instruction (CMD or ENTRYPOINT)
    labels_to_add = []
    if not has_framework_label:
        labels_to_add.append(f'LABEL framework="{framework}"\n')
    if not has_dataset_label:
        labels_to_add.append(f'LABEL dataset="{dataset}"\n')
    if not has_model_label:
        labels_to_add.append(f'LABEL model="{model}"\n')
    if not has_batch_size_label:
        labels_to_add.append(f'LABEL batch_size={batch_size}\n')
    if not has_learning_rate_label:
        labels_to_add.append(f'LABEL learning_rate={learning_rate}\n')
    if not has_num_epochs_label:
        labels_to_add.append(f'LABEL num_epochs={num_epochs}\n')

    if labels_to_add:
        # Insert the labels before the last command
        for i in range(len(dockerfile_lines) - 1, -1, -1):
            if dockerfile_lines[i].startswith(('CMD', 'ENTRYPOINT')):
                dockerfile_lines = dockerfile_lines[:i] + labels_to_add + dockerfile_lines[i:]
                break
        else:
            # If no CMD or ENTRYPOINT is found, append the labels at the end
            dockerfile_lines.extend(labels_to_add)

        # Write back the updated Dockerfile
        with open(dockerfile_path, 'w') as file:
            file.writelines(dockerfile_lines)
        print(f"Updated Dockerfile with missing labels: {', '.join([label.split()[1] for label in labels_to_add])}")
    else:
        print("Dockerfile already contains the required labels.")

# Get the directory to search from the command line argument
if len(sys.argv) != 2:
    print("Usage: python3 app.py <directory>")
    sys.exit(1)

directory = sys.argv[1]
config_file = find_config_file(directory)
dockerfile = find_dockerfile(directory)

if config_file:
    framework, dataset, model, batch_size, learning_rate, num_epochs = extract_workload_from_config(config_file)
    if framework and dataset and model and batch_size is not None and learning_rate is not None and num_epochs is not None:
        print(f"framework: {framework}")
        print(f"dataset: {dataset}")
        print(f"model: {model}")
        print(f"batch_size: {batch_size}")
        print(f"learning_rate: {learning_rate}")
        print(f"num_epochs: {num_epochs}")

        if dockerfile:
            check_and_update_dockerfile(dockerfile, framework, dataset, model, batch_size, learning_rate, num_epochs)
        else:
            print(f"Dockerfile not found in directory: {directory}")
    else:
        print("One or more fields (framework, dataset, model, batch_size, learning_rate, num_epochs) not found in the config.ini")
else:
    print(f"config.ini not found in directory: {directory}")

