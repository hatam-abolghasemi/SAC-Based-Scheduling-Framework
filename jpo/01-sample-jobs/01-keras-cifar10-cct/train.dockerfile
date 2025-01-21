FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Link dataset from dataset container
COPY --from=cifar10:1.0.3 /app/data /data

ENTRYPOINT ["python", "train.py"]

