#############
# Variables #
#############
DOCKERFILE_PATH=./Dockerfile
DATASET_PATH=./data/chopstick.csv

# different ways of creating TensorFlow models require usage of different APIs 
# to work with TensorFlow Serving. Supported values are estimator_api and tensorflow_api
ifndef API_TO_USE
API_TO_USE=estimator_api
endif

CLASSIFIER_SCRIPT=./$(API_TO_USE)/chopstick_classifier.py

# servables will be exported to this directory after model traning completes
SERVABLES_PATH=$(CURDIR)/serving

#############
# Tasks     #
#############
clean:
	rm -rf ./serving	

tfserve_image: $(DOCKERFILE_PATH)
	docker build . -f $(DOCKERFILE_PATH) -t tfserve_bin

train_classifier: $(DATASET_PATH) $(CLASSIFIER_SCRIPT)
	python $(CLASSIFIER_SCRIPT) $(DATASET_PATH) --val-num=20

run_server: $(SERVABLES_PATH) 
	docker run -p8500:8500 -d --rm -v $(SERVABLES_PATH):/models tfserve_bin
