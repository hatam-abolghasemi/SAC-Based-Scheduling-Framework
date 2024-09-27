import os
import re

# Define the expected keys and their types
expected_keys_types = {
    "positional_emb": "bool",
    "conv_layers": "int",
    "projection_dim": "int",
    "num_heads": "int",
    "transformer_units": "list",
    "transformer_layers": "int",
    "stochastic_depth_rate": "float",
    "learning_rate": "float",
    "weight_decay": "float",
    "batch_size": "int",
    "num_epochs": "int",
    "image_size": "int",
    "num_classes": "int",
    "input_shape": "tuple"
}

# Function to check if a string is a valid integer
def is_integer(value):
    return re.match(r'^-?[0-9]+$', value) is not None

# Function to check if a string is a valid float
def is_float(value):
    return re.match(r'^-?[0-9]+(\.[0-9]+)?$', value) is not None

# Function to check if a string is a valid boolean
def is_boolean(value):
    return value in ["True", "False"]

# Function to check if a string is a valid tuple
def is_tuple(value):
    return re.match(r'^[0-9]+(,[0-9]+)*$', value) is not None

# Function to check if a string is a valid list of integers
def is_list_of_integers(value):
    nums = value.split(',')
    return all(is_integer(num) for num in nums)

# Initialize variables
config_file = ""
app_file = ""
total_errors = 0

# Step 1: Search for config.ini files in subdirectories
for root, dirs, files in os.walk('.'):
    for file in files:
        if file == 'config.ini':
            config_file = os.path.join(root, file)
            break
    if config_file:
        break

# Check if a config file was found
if not config_file:
    print("INFO: Found no config.ini file anywhere.")
else:
    print(f"INFO: Found a config.ini file at location {config_file}.")

# Step 2: Search for a Python app file (app.py)
for root, dirs, files in os.walk('.'):
    for file in files:
        if file == 'app.py':
            app_file = os.path.join(root, file)
            break
    if app_file:
        break

# Check if a Python app file was found
if not app_file:
    print("INFO: Found no Python app anywhere.")
else:
    print(f"INFO: Found a Python app at location {app_file}.")

# Step 3: Check if the corresponding app.py uses the config file
if config_file and app_file:
    with open(app_file, 'r') as f:
        app_content = f.read()
    if os.path.basename(config_file) in app_content:
        print("INFO: The found Python app uses the found config file.")
    else:
        print("INFO: The found Python app doesn't use the found config file.")

# If no config file or app file was found, exit
if not config_file or not app_file:
    exit(0)

# Step 4: Read and validate the config file content
with open(config_file, 'r') as f:
    for line in f:
        # Remove leading and trailing whitespace
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Skip section headers
        if re.match(r'^\[.*\]$', line):
            continue

        # Split the line into key and value
        key, value = map(str.strip, line.split('=', 1))

        # Check if the key is recognized
        if key not in expected_keys_types:
            print(f"INFO: Skipping unrecognized key '{key}'.")
            continue

        # Validate the value against its expected type
        expected_type = expected_keys_types[key]
        if expected_type == "int":
            if not is_integer(value):
                print(f"ERROR: Invalid type for '{key}' in {config_file}. Expected int, got '{value}'.")
                total_errors += 1
        elif expected_type == "float":
            if not is_float(value):
                print(f"ERROR: Invalid type for '{key}' in {config_file}. Expected float, got '{value}'.")
                total_errors += 1
        elif expected_type == "bool":
            if not is_boolean(value):
                print(f"ERROR: Invalid type for '{key}' in {config_file}. Expected bool, got '{value}'.")
                total_errors += 1
        elif expected_type == "tuple":
            if not is_tuple(value):
                print(f"ERROR: Invalid type for '{key}' in {config_file}. Expected tuple, got '{value}'.")
                total_errors += 1
        elif expected_type == "list":
            if not is_list_of_integers(value):
                print(f"ERROR: Invalid type for '{key}' in {config_file}. Expected list of integers, got '{value}'.")
                total_errors += 1

# Final report
if total_errors > 0:
    print(f"ERROR: Total errors across all config files: {total_errors}.")
else:
    print("INFO: No errors found. The config file is valid.")
