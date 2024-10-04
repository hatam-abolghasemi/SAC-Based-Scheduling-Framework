FROM python:3.9-slim
WORKDIR /app

COPY app.py .
COPY config.ini .
COPY requirements.txt .
RUN pip install -r requirements.txt

LABEL framework="keras"
LABEL dataset="cifar10"
LABEL model="cct"

CMD ["python", "app.py"]
