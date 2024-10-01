import os
import configparser
import sys

# Set the config.ini file name
CONFIG_FILE = "config.ini"

def extract_workload_from_config(config_file):
    """Extract the framework, dataset, and model from the [WORKLOAD] section in config.ini."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return (
        config.get('WORKLOAD', 'framework', fallback=None),
        config.get('WORKLOAD', 'dataset', fallback=None),
        config.get('WORKLOAD', 'model', fallback=None)
    )

def find_config_file(directory):
    """Find config.ini in the specified directory."""
    config_path = os.path.join(directory, CONFIG_FILE)
    return config_path if os.path.exists(config_path) else None

# Get the directory to search from the command line argument
if len(sys.argv) != 2:
    print("Usage: python3 01-workload-detector.py <directory>")
    sys.exit(1)

directory = sys.argv[1]
config_file = find_config_file(directory)

if config_file:
    framework, dataset, model = extract_workload_from_config(config_file)
    if framework and dataset and model:
        print(f"framework: {framework}")
        print(f"dataset: {dataset}")
        print(f"model: {model}")
    else:
        print("One or more fields (framework, dataset, model) not found in the config.ini")
else:
    print(f"config.ini not found in directory: {directory}")

