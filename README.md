# Awesome Chopstick Classifier. TensorFlow Serving examples for Kubernetes
This project proveides examples on different ways to train, export and serve 
TensorFlow models using Docker, Kubernetes, and TensorFlow Serving.
This repository contains conpanion code for the tutorial LINK TBD.

## Prerequisites
* [Docker](https://www.docker.com)
* [Kubernetes](https://kubernetes.io). Consider using [Minikube](https://github.com/kubernetes/minikube) for local experiments and [kubeadm](https://kubernetes.io/docs/setup/independent/create-cluster-kubeadm/) for simple cluster set up.
* For model training and export: Python 3, `pip install tensorflow colorama`
* For running the client: Python 2, `pip install grpcio tensorflow tensorflow-serving-api colorama`

## Building and running the TensorFlow Serving image
You may use provided Makefile to 
* Build the image with `make tfserve_image`. Notice that no compilation isperformed through the process, the image uses precompiled TensorFlow Serving Ubuntu package.
* Train the classifier and export it with `make train_classifier`
* Run TensorFlow Serving via docker with `make run_server`

## Using different APIs
TensorFlow Serving examples are provided both for TensorFlow Estimator API and 
for plain TensorFlow clode since approaches for model export differ 
significantly between those.

You can use `API_TO_USE` variable to set the API of interest (the default one
is `estimator_api`):
```
make API_TO_USE=tf_api clean train_classifier
make API_TO_USE=estimator_api clean train_classifier
```

## Calling example client
You must use python 2 on the client side as the `tensorflow-serving-api` [package](https://pypi.python.org/pypi/tensorflow-serving-api/) is only provided for Python 2 at the current moment. You may call the example client using this command:
```
python2 client.py --model-name tf_model localhost:8500 0.5
```

Execute `python2 client.py -h` if you want to kow more about the parameters.

