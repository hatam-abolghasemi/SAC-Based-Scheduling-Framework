# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
RUN pip install torch
RUN pip install torchvision

COPY . /app

# Set environment variable for dataset
ENV DATAROOT=/app/data

# Copy the data from the dataset container (assuming dataset container's volume is mounted to /app/data)
COPY --from=dcgan-dataset:1.0.1 /app/data /app/data

LABEL framework="pytorch"
LABEL dataset="cifar10"
LABEL model="dcgan"
LABEL batch_size=64
LABEL learning_rate=0.0002
LABEL num_epochs=25

# Set the entrypoint to run the training script
CMD ["python", "train.py"]

