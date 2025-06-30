import os
import numpy as np
import keras

# Set up proxy for the dataset download
os.environ["http_proxy"] = "http://172.16.2.190:8118"
os.environ["https_proxy"] = "http://172.16.2.190:8118"

# Directory to save the dataset
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

def download_dataset():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Save the dataset to files in the specified directory
    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "x_test.npy"), x_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    print("Dataset downloaded and saved to the data directory.")

# Download and save the dataset
download_dataset()

