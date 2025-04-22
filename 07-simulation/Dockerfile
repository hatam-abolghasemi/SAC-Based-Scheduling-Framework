# Use the official lightweight Python 3.9 image
FROM python:3.9-slim

# Set environment variables to avoid prompts during installation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required for some of the Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the application files into the container (assuming the application is in the same directory as this Dockerfile)
COPY . .

# Command to run the application
CMD ["python", "prometheus-reloader.py"]

