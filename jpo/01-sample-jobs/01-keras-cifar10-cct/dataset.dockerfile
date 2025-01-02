FROM python:3.9-slim

WORKDIR /app
COPY dataset.py /app/dataset.py
RUN pip install --no-cache-dir keras tensorflow
RUN mkdir -p data
RUN ["python", "dataset.py"]

