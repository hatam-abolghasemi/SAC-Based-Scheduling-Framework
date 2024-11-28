FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY config.ini .

LABEL framework="keras"
LABEL dataset="cifar10"
LABEL model="cct"
LABEL batch_size=128
LABEL learning_rate=0.001
LABEL num_epochs=15

CMD ["python", "app.py"]
