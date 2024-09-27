import os
import re
import configparser

# Set the files to search for
SEARCH_FILE = "app.py"
CONFIG_FILE = "config.ini"

def search_file_content(file_path, pattern):
    """Search for a pattern in a specified file."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return re.search(pattern, content) is not None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False

def extract_dataset(file_path):
    """Extract dataset information from app.py."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            match = re.search(r'keras\.datasets\.([a-zA-Z0-9_]+)\.load_data\(\)', content)
            if match:
                return match.group(1)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return None

def extract_model_from_config(config_file):
    """Extract the model name from the config.ini."""
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        return config.get('DATA', 'model', fallback=None)
    except configparser.Error:
        print(f"Error reading config file: {config_file}")
        return None

def find_files_in_subdirectories():
    """Find app.py and config.ini in every subdirectory."""
    app_paths = []
    config_paths = []

    for root, dirs, files in os.walk('.'):  # '.' means current directory
        if SEARCH_FILE in files:
            app_paths.append(os.path.join(root, SEARCH_FILE))
        if CONFIG_FILE in files:
            config_paths.append(os.path.join(root, CONFIG_FILE))
    
    return app_paths, config_paths

# Find all app.py and config.ini files in subdirectories
app_files, config_files = find_files_in_subdirectories()

# Check for imports and print the frameworks found
for app_file in app_files:
    if search_file_content(app_file, r'import\s+keras'):
        print("Found framework: keras")
        
        # Look for keras datasets usage
        dataset = extract_dataset(app_file)
        if dataset:
            print(f"Found dataset: {dataset}")

    if search_file_content(app_file, r'import\s+tensorflow'):
        print("Found framework: tensorflow")

    if search_file_content(app_file, r'import\s+scikit-learn'):
        print("Found framework: scikit-learn")

    if search_file_content(app_file, r'import\s+torch'):
        print("Found framework: pytorch")

# Check for model name in each config.ini found
for config_file in config_files:
    model = extract_model_from_config(config_file)
    if model:
        print(f"Found model: {model}")
