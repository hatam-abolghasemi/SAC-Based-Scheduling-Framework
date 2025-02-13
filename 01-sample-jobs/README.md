## How to run these sample jobs?

* Go to each directory and cat the `train.dockerfile` and find the dataset image name. Then:
```
docker build -t <dataset-image> -f dataset.dockerfile .
```

* Then check the `docker-compose.yml` at here to tag the ready-to-train images:
```
docker build -t <training-image> -f train.dockerfile .
```

* At last you can run them simultaneously:
```
docker compose up -d
```
