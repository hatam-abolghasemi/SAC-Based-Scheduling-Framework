FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

LABEL framework="pytorch"
LABEL dataset="cartpole"
LABEL model="actor-critic"

CMD ["python", "app.py"]
