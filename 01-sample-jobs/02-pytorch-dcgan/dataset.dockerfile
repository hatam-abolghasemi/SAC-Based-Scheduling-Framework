# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install torchvision

# Set the environment variable for the dataset directory
ENV DATAROOT=/app/data

# Run the dataset download script
RUN ["python", "dataset.py", "--dataroot", "/app/data"]

