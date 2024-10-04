# Job Detector

This Python application automatically extracts workload parameters from a `config.ini` file and updates the corresponding Dockerfile to ensure it contains the necessary labels for the framework, dataset, and model.

## Table of Contents
* Usage
* Code Overview
* Execution Flow

###  Usage
To run the application, execute the following command in your terminal, specifying the directory containing the `config.ini` and `Dockerfile`:

bash
```
python3 app.py <directory>

# Example
python3 app.py /path/to/your/directory
```

### Code Overview
The application consists of the following key functions:

* `extract_workload_from_config(config_file)`: Extracts the framework, dataset, and model from the [WORKLOAD] section of the `config.ini` file.
* `find_config_file(directory)`: Searches for the `config.ini` file in the specified directory.
* `find_dockerfile(directory)`: Searches for the `Dockerfile` in the specified directory.
* `check_and_update_dockerfile(dockerfile_path, framework, dataset, model)`: Checks if the Dockerfile contains the necessary labels for framework, dataset, and model. If any labels are missing, they are added before the last command in the `Dockerfile`.

### Execution Flow
* The user provides a directory containing the `config.ini` and `Dockerfile`.
* The application checks for the existence of the `config.ini` file.
    * If found, it extracts the framework, dataset, and model.
* It then checks for the existence of the `Dockerfile`.
    * If the `Dockerfile` is found, it verifies and updates it with any missing labels.
