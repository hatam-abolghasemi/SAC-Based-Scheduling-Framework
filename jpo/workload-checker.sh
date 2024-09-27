#!/bin/bash

# Set the files to search in
SEARCH_FILE="app.py"  # Change this to your actual file name if different
CONFIG_FILE="config.ini"

# Check for imports and print the framework found
if grep -q "import keras" "$SEARCH_FILE"; then
    echo "Found framework: keras"

    # Look for keras datasets usage
    if grep -q "keras.datasets." "$SEARCH_FILE"; then
        DATASET=$(grep "keras.datasets." "$SEARCH_FILE" | sed -E 's/.*keras\.datasets\.([a-zA-Z0-9_]+)\.load_data().*/\1/')
        echo "Found dataset: $DATASET"
    fi
fi

if grep -q "import tensorflow" "$SEARCH_FILE"; then
    echo "Found framework: tensorflow"
fi

if grep -q "import scikit-learn" "$SEARCH_FILE"; then
    echo "Found framework: scikit-learn"
fi

if grep -q "import torch" "$SEARCH_FILE"; then
    echo "Found framework: pytorch"
fi

# Check for model name in config.ini
if grep -q "model=" "$CONFIG_FILE"; then
    MODEL=$(grep "model=" "$CONFIG_FILE" | sed 's/model=//')
    echo "Found model: $MODEL"
fi
