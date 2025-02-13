FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
COPY app-k8s.py /app
EXPOSE 4223

CMD ["python", "app-k8s.py"]

