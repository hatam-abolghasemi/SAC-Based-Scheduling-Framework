
* No need to use the local registry on 5001. You can load images like this:
```
sudo kind load docker-image gcr.io/google-samples/hello-app:1.0 --name kind-cluster --nodes kind-cluster-worker3,kind-cluster-worker2,kind-cluster-worker
```
