FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gym==0.21

COPY app.py .

CMD ["python", "app.py"]
