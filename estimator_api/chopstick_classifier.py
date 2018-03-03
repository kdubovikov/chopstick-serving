"""Train chopstick efficiency classifier and export it as Servable"""
import argparse

import pandas as pd
import tensorflow as tf


def preprocess_data(filename, val_rows_count=20):
    """Load data and split it to train and validation sets"""

    # We will read and process data with pandas. This can be done directly
    # via tensorflow, but pandas is way more convenient to deal with.
    # You may consider using tensorflow queued loaders with large datasets
    # that do not fit in memory
    data = pd.read_csv(filename, sep=',')

    # We split food pincing efficiency to 3 bins of equal size based on the
    # range present in traning data. This way problem is converted from
    # regression with continuous optput to classification with
    # the number of classes equal to the number of bins
    data['Food.Pinching.Efficiency'] = pd.cut(
        data['Food.Pinching.Efficiency'], bins=3, labels=[0, 1, 2]).astype(int)

    data_train = data[:-val_rows_count]
    data_val = data[-val_rows_count:]

    x_train = data_train.drop('Food.Pinching.Efficiency', axis=1)
    y_train = data_train['Food.Pinching.Efficiency']
    x_val = data_val.drop('Food.Pinching.Efficiency', axis=1)
    y_val = data_val['Food.Pinching.Efficiency']

    return x_train, x_val, y_train, y_val


def input_fn(x_df, y_df):
    """Transform pandas dataframes to input format suitable for
    tensorflow estimator API"""
    return tf.estimator.inputs.pandas_input_fn(
        x=x_df,
        y=y_df,
        batch_size=100,
        num_epochs=2,
        shuffle=True,
        num_threads=5)


def train_estimator(x_train, y_train):
    """Train linear model on the data"""
    # Number of classes can be inferred automatically
    n_classes = y_train.unique().shape[0]
    chopstick_len = tf.feature_column.numeric_column('Chopstick.Length')
    estimator = tf.estimator.LinearClassifier(feature_columns=[chopstick_len],
                                              n_classes=n_classes)

    estimator.train(input_fn=input_fn(x_train, y_train))

    return estimator


def eval_estimator(est, x_val, y_val):
    """Evaluate estimator and print the results"""
    results = est.evaluate(input_fn=input_fn(x_val, y_val), steps=None)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""

    # feature spec dictionary  determines our input parameters for the model
    feature_spec = {
        'Chopstick.Length': tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }

    # the inputs will be initially fed as strings with data serialized by
    # Google ProtoBuffers
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[1], name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}

    # deserialize input
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--val-num', type=int, default=20, dest='val_num',
                        help='number of examples to use as a validation set')
    args = parser.parse_args()

    x_train, x_val, y_train, y_val = preprocess_data(args.filename,
                                                     val_rows_count=args.val_num)

    est = train_estimator(x_train, y_train)
    eval_estimator(est, x_val, y_val)
    est.export_savedmodel('./serving', serving_input_receiver_fn)


if __name__ == "__main__":
    main()
