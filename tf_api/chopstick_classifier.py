"""Train chopstick efficiency classifier and export it as Servable"""
import argparse

import numpy as np
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
        data['Food.Pinching.Efficiency'], bins=3, labels=[0, 1, 2]).astype(float)

    data_train = data[:-val_rows_count]
    data_val = data[-val_rows_count:]

    x_train = data_train.drop(['Food.Pinching.Efficiency', 'Individual'], axis=1)
    y_train = data_train['Food.Pinching.Efficiency']
    x_val = data_val.drop(['Food.Pinching.Efficiency', 'Individual'], axis=1)
    y_val = data_val['Food.Pinching.Efficiency']
    
    # One hot encoding for labels
    y_train = pd.get_dummies(y_train)
    y_val = pd.get_dummies(y_val)

    return x_train, x_val, y_train, y_val


class LogisticRegression:
    """Simple implementation of logistic regression with plain tensorflow"""

    def __init__(self, num_features, learning_rate):
        # Let's create input placeholders
        self.x = tf.placeholder(tf.float32, [None, num_features])
        self.y = tf.placeholder(tf.int32, [None, 3])

        # Weight and bias variables  
        W = tf.Variable(tf.truncated_normal([num_features, 3]),
                        [num_features, 3],
                        dtype=tf.float32)
        b = tf.Variable(tf.zeros([3]))

        # Calculate unnormalized target estimates
        y_hat = tf.matmul(self.x, W) + b

        # Optimize with SGD
        self.loss = tf.losses.softmax_cross_entropy(self.y, y_hat)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        
        # Operations for training and inference
        self.optimize_op = optimizer.minimize(self.loss)
        self.predict_op = tf.nn.softmax(y_hat)

    def train(self, x, y):
        sess = tf.get_default_session()
        _, loss = sess.run([self.optimize_op, self.loss], feed_dict={self.x: x, self.y: y})
        return loss

    def evaluate(self, x, y):
        """Evaluate trained model on given inputs

        Returns
        -------
        accuracy: float
            prediction accuracy
        """

        sess = tf.get_default_session()
        preds = sess.run(self.predict_op, feed_dict={self.x: x, self.y: y})

        preds = np.asarray(preds)
        y = np.asarray(y)
        return (preds == y).all(axis=1).sum() / len(preds)


    def export_for_serving(self, model_path):
        # SavedModelBuilder will perform model export for us
        builder = tf.saved_model.builder.SavedModelBuilder(model_path + '/1')

        # First, we need to describe the signature for our API.
        # It will consist of single prediction method with chopstck_length as 
        # an input and class probability as an output.
        # We build TensorInfo protos as a starting step. Those are needed to 
        # shape prediction method signature
        tensor_info_x = tf.saved_model.utils.build_tensor_info(self.x)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(self.predict_op)

        prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'chopstick_length': tensor_info_x},
            outputs={'classes_prob': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        
        # Legacy calls to initialize tables
        # For more details on this see
        # https://stackoverflow.com/questions/45521499/legacy-init-op-in-tensorflow-serving
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')

        # Finally, let's export the model as servable!
        # We use DEFAULT_SERVING_SIGNATURE_DEF_KEY flag to mark our
        # prediction_signature as a default method to call
        sess = tf.get_default_session()
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--val-num', type=int, default=20, dest='val_num',
                        help='number of examples to use as a validation set')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=100, dest='batch_size',
                        help='training batch size')
    args = parser.parse_args()
    
    # Preprocess data
    x_train, x_val, y_train, y_val = preprocess_data(args.filename,
                                                     val_rows_count=args.val_num)

    print(x_train.shape)
    print(x_train.head())
    
    # Create estimator
    est = LogisticRegression(x_train.shape[1], 0.01)

    # Perform model training and export it for serving
    with tf.Session().as_default() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(0, args.epochs):
            print(f"Epoch {epoch}")

            for batch_idx in range(0, len(x_train), len(x_train) %
                    args.batch_size):

                # Get the next training batch
                if batch_idx + args.batch_size <= len(x_train):
                    next_batch_idx = batch_idx + args.batch_size     
                else:
                    next_batch_idx = len(x_train)

                batch_x = x_train[batch_idx:next_batch_idx]
                batch_y = y_train[batch_idx:next_batch_idx]

                # Perform training step
                loss = est.train(batch_x, pd.get_dummies(batch_y))
                print(f"Batch loss = {loss:3.4}")

        print(f"Validation accuracy = {est.evaluate(x_val, y_val)}")

        print("Exporting model...")
        est.export_for_serving('./serving')
        print("Done!")


if __name__ == "__main__":
    main()
