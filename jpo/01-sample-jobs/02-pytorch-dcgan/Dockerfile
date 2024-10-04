FROM python:3.9-slim
WORKDIR /app

COPY app.py .
COPY config.ini .
COPY requirements.txt .
RUN pip install -r requirements.txt

LABEL framework="pytorch"
LABEL dataset="lsun"
LABEL model="dcgan"
CMD ["python", "app.py", "--dataset", "cifar10", "--dataroot", "."]
