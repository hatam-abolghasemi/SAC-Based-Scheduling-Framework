FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Link dataset from dataset container
COPY --from=cifar10:1.0.3 /app/data /data

LABEL framework="keras"
LABEL dataset="cifar10"
LABEL model="cct"
LABEL batch_size=128
LABEL learning_rate=0.001
LABEL num_epochs=15

ENTRYPOINT ["python", "train.py"]

