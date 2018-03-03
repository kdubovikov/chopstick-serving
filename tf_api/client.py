from __future__ import absolute_import, division, print_function

import argparse

import tensorflow as tf
from colorama import Back, Fore, Style, init
from grpc.beta import implementations

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

init()


def create_request(data, request):
    print(data)
    chopstick_length = tf.contrib.util.make_tensor_proto(data,
            dtype=tf.float32, shape=[1, 1])
    request.inputs['chopstick_length'].CopyFrom(chopstick_length)
    return request


def main():
    parser = argparse.ArgumentParser(description='CTR model gRPC client')

    parser.add_argument(
        'tf_server',
        type=str,
        help='host:port for CTR Model TendorFlow Server')

    parser.add_argument(
        'chopstick_length',
        type=float,
        help='chopstick length to classify')

    parser.add_argument(
        '--model-name',
        type=str,
        default='tf_model',
        dest='model_name',
        help='model name to use')

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        default=False,
        dest='verbose',
        help='verbose output')

    args = parser.parse_args()

    host, port = args.tf_server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    
    # We use predefined ClassificationRequest protobuf here. This API is useful
    # when you are working with tf.estimator package.
    # tf.estimator.export.ServingInputReceiver expects to revieve TFExample
    # serialized into string. All serialization and deserialization code is
    # handled for us with gRPC and ClassificationRequest/ServingInputReceiver
    # API.
    #
    # Consider using tf.make_tensor_proto function and
    # tf.saved_model.builder.SavedModelBuilder if you are not using
    # tf.estimator API
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args.model_name

    request_data = create_request(args.chopstick_length, request)

    print("Sending request")
    result_future = stub.Predict(request_data, 20.0)  # 5 seconds
    print(result_future)

def response_callback(result_future):
    exception = result_future.exception()

    if exception:
        print(Fore.RED + Style.BRIGHT + "Exception from TF Server: %s" % exception)
        return

    result = result_future.result().outputs
    print(result)


if __name__ == "__main__":
    main()
