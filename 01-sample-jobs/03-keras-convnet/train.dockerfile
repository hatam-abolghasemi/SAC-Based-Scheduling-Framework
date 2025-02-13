FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py .

# Link dataset from dataset container
COPY --from=keras-convnet-dataset:1.0.0 /app/data /data

LABEL framework="keras"
LABEL dataset="mnist"
LABEL model="convnet"
LABEL batch_size=128
LABEL num_epochs=15

ENTRYPOINT ["python", "train.py"]

