import numpy as np
import keras
import os

# Ensure the 'data' directory exists
output_dir = '/app/data'
os.makedirs(output_dir, exist_ok=True)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Save the dataset as .npy files
np.save(f'{output_dir}/x_train.npy', x_train)
np.save(f'{output_dir}/y_train.npy', y_train)
np.save(f'{output_dir}/x_test.npy', x_test)
np.save(f'{output_dir}/y_test.npy', y_test)

print(f"Dataset saved at {output_dir} directory.")

