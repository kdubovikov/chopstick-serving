FROM ubuntu:16.04
RUN "sh" "-c" "echo nameserver 8.8.8.8 >> /etc/resolv.conf"
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
	apt-get update && apt-get install -y \
        curl \
        git \
        python-dev \
        python-numpy \
        python-pip \
        libcurl3-dev \
        && \
	curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - && \
	apt-get update && \
	apt-get install -y tensorflow-model-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up grpc

RUN pip install mock grpcio

VOLUME /models
CMD ["tensorflow_model_server", "--model_name=tf_model", "--model_base_path=/models"]
