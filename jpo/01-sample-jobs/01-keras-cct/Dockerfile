FROM python:3.9-slim
WORKDIR /app

COPY app.py .
COPY config.ini .
COPY requirements.txt .
RUN pip install -r requirements.txt

LABEL framework="keras"
LABEL dataset="cifar10"
LABEL model="cct"

LABEL batch_size=128
LABEL learning_rate=0.001
LABEL num_epochs=30
CMD ["python", "app.py"]
