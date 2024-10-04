FROM python:3.9-slim
WORKDIR /app

COPY app.py .
COPY config.ini .
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && \
    apt-get install -y curl && \
    curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
RUN tar -xzf cifar-10-python.tar.gz

CMD ["python", "app.py"]
