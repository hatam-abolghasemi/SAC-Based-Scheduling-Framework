import os
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical

# Dataset path from the linked dataset container
data_dir = "/data"

# Check if the dataset exists
if not all(os.path.exists(os.path.join(data_dir, fname)) for fname in ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]):
    raise FileNotFoundError("Dataset not found in /data. Ensure the dataset container is properly linked.")

# Load the dataset
x_train = np.load(os.path.join(data_dir, "x_train.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
x_test = np.load(os.path.join(data_dir, "x_test.npy"))
y_test = np.load(os.path.join(data_dir, "y_test.npy"))

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert class vectors to binary class matrices
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Print dataset information
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"{x_train.shape[0]} training samples")
print(f"{x_test.shape[0]} test samples")

# Build the model
model = Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.summary()

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
batch_size = 128
epochs = 15
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the trained model
model.save("trained_model.h5")
print("Model saved to /app/trained_model.h5")

