#!/bin/bash

# Define the expected keys and their types
declare -A expected_keys_types
expected_keys_types=(
    ["positional_emb"]="bool"
    ["conv_layers"]="int"
    ["projection_dim"]="int"
    ["num_heads"]="int"
    ["transformer_units"]="list"
    ["transformer_layers"]="int"
    ["stochastic_depth_rate"]="float"
    ["learning_rate"]="float"
    ["weight_decay"]="float"
    ["batch_size"]="int"
    ["num_epochs"]="int"
    ["image_size"]="int"
    ["num_classes"]="int"
    ["input_shape"]="tuple"
)

# Function to check if a string is a valid integer
is_integer() {
    [[ "$1" =~ ^-?[0-9]+$ ]]
}

# Function to check if a string is a valid float
is_float() {
    [[ "$1" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]
}

# Function to check if a string is a valid boolean
is_boolean() {
    [[ "$1" == "True" || "$1" == "False" ]]
}

# Function to check if a string is a valid tuple
is_tuple() {
    [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]
}

# Function to check if a string is a valid list of integers
is_list_of_integers() {
    IFS=',' read -ra nums <<< "$1"
    for num in "${nums[@]}"; do
        if ! is_integer "$num"; then
            return 1
        fi
    done
    return 0
}

# Initialize variables
config_file=""
app_file=""
total_errors=0

# Step 1: Search for config.ini files in subdirectories
while IFS= read -r -d '' file; do
    config_file="$file"
done < <(find . -name 'config.ini' -print0)

# Check if a config file was found
if [[ -z "$config_file" ]]; then
    echo "INFO: Found no config.ini file anywhere."
else
    echo "INFO: Found a config.ini file at location $config_file."
fi

# Step 2: Search for a Python app file (app.py)
while IFS= read -r -d '' file; do
    app_file="$file"
done < <(find . -name 'app.py' -print0)

# Check if a Python app file was found
if [[ -z "$app_file" ]]; then
    echo "INFO: Found no Python app anywhere."
else
    echo "INFO: Found a Python app at location $app_file."
fi

# Step 3: Check if the corresponding app.py uses the config file
if [[ -n "$config_file" && -n "$app_file" ]]; then
    if grep -q "$(basename "$config_file")" "$app_file"; then
        echo "INFO: The found Python app uses the found config file."
    else
        echo "INFO: The found Python app doesn't use the found config file."
    fi
fi

# If no config file or app file was found, exit
if [[ -z "$config_file" || -z "$app_file" ]]; then
    exit 0
fi

# Step 4: Read and validate the config file content
while IFS='=' read -r key value; do
    # Remove leading and trailing whitespace
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)

    # Skip empty lines and comments
    [[ -z "$key" || "$key" =~ ^# ]] && continue

    # Skip section headers
    if [[ "$key" =~ ^\[.*\]$ ]]; then
        continue
    fi

    # Check if the key is recognized
    if [[ ! ${expected_keys_types[$key]} ]]; then
        echo "INFO: Skipping unrecognized key '$key'."
        continue
    fi

    # Validate the value against its expected type
    case "${expected_keys_types[$key]}" in
        "int")
            if ! is_integer "$value"; then
                echo "ERROR: Invalid type for '$key' in $config_file. Expected int, got '$value'."
                ((total_errors++))
            fi
            ;;
        "float")
            if ! is_float "$value"; then
                echo "ERROR: Invalid type for '$key' in $config_file. Expected float, got '$value'."
                ((total_errors++))
            fi
            ;;
        "bool")
            if ! is_boolean "$value"; then
                echo "ERROR: Invalid type for '$key' in $config_file. Expected bool, got '$value'."
                ((total_errors++))
            fi
            ;;
        "tuple")
            if ! is_tuple "$value"; then
                echo "ERROR: Invalid type for '$key' in $config_file. Expected tuple, got '$value'."
                ((total_errors++))
            fi
            ;;
        "list")
            if ! is_list_of_integers "$value"; then
                echo "ERROR: Invalid type for '$key' in $config_file. Expected list of integers, got '$value'."
                ((total_errors++))
            fi
            ;;
    esac
done < <(grep -v '^\s*#' "$config_file") # Ignore comments

# Final report
if (( total_errors > 0 )); then
    echo "ERROR: Total errors across all config files: $total_errors."
else
    echo "INFO: No errors found. The config file is valid."
fi
