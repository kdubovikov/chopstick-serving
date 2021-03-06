from __future__ import absolute_import, division, print_function

import argparse

import tensorflow as tf
from colorama import Back, Fore, Style, init
from grpc.beta import implementations

from tensorflow_serving.apis import classification_pb2, prediction_service_pb2

init()


def create_request(data, request):
    indicies = tf.contrib.util.make_tensor_proto(data[0], dtype=tf.int64)
    request.inputs['input-indices'].CopyFrom(indicies)
    values = tf.contrib.util.make_tensor_proto(data[1], dtype=tf.float32)
    request.inputs['input-values'].CopyFrom(values)
    shape = tf.contrib.util.make_tensor_proto(data[2], dtype=tf.int64)
    request.inputs['input-shape'].CopyFrom(shape)
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
    request = classification_pb2.ClassificationRequest()
    request.model_spec.name = args.model_name
    example = request.input.example_list.examples.add()
    example.features.feature['Chopstick.Length'].float_list.value\
                                .append(args.chopstick_length)

    result = stub.Classify(request, 10.0)  # 10 secs timeout 

    print(result)

def response_callback(result_future):
    exception = result_future.exception()

    if exception:
        print(Fore.RED + Style.BRIGHT + "Exception from TF Server: %s" % exception)
        return

    result = result_future.result().outputs
    print(result)


if __name__ == "__main__":
    main()
