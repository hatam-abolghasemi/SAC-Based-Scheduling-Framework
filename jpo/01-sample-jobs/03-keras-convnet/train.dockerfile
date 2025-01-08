FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py .

# Link dataset from dataset container
COPY --from=keras-convnet-dataset:1.0.0 /app/data /data

ENTRYPOINT ["python", "train.py"]

