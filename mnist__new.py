from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import time

# Deopout rate
RATE_DROPOUT = 0.5

def small_cnn(x, phase_train):
    # Dense Layer
    pool2_flat = tf.reshape(x, [-1, 4 * 4 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=RATE_DROPOUT, training=phase_train)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    logits = tf.layers.dropout(inputs=logits, rate=RATE_DROPOUT, training=phase_train)

    return logits

def wrap(x, m, n, stride, shape):
    slicing = tf.TensorArray('float32', m * n)
    for j in range(m):
        for k in range(n):
            slicing = slicing.write(
                j * n + k, tf.slice(x, [0, j * stride, k * stride, 0],
                            shape))
    sliced = tf.reshape(slicing.concat(), shape)
    slicing.close().mark_used()
    return sliced


def model(x):
    phase_train = tf.placeholder(tf.bool)
    m = 5
    n = 5
    stride = 3

    x = tf.reshape(x, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    conv1_dropout = tf.layers.dropout(inputs=conv1, rate=RATE_DROPOUT, training=phase_train)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1_dropout,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    conv2_dropout = tf.layers.dropout(inputs=conv2, rate=RATE_DROPOUT, training=phase_train)

    # Pooling Layer #1
    pool1 = tf.layers.average_pooling2d(inputs=conv2_dropout, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    conv3_dropout = tf.layers.dropout(inputs=conv3, rate=RATE_DROPOUT, training=phase_train)

    # Pooling Layer #2
    pool2 = tf.layers.average_pooling2d(inputs=conv3_dropout, pool_size=[2, 2], strides=2)

    padding = tf.pad(pool2, [[0, 0], [1, 1], [1, 1], [0, 0]])
    logits = small_cnn(wrap(padding, 6, 6, 1, [-1, 4, 4, 64]), phase_train)
    logits = tf.reduce_mean(tf.reshape(logits, [36, -1, 10]), 0)

    return logits, phase_train


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    input_data = tf.placeholder(tf.float32, [None, 784])
    output_data = tf.placeholder(tf.int64, [None])
    y_model, phase_train= model(input_data)

    #Loss
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=output_data, logits=y_model)
    cross_entropy = tf.reduce_mean(cross_entropy)

    #Optimizer
    rate = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)

    #Accuracy
    correct_prediction = tf.equal(tf.argmax(y_model, 1), output_data)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    #Congifg
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        t0 = time.clock()
        rt = 1e-3
        for i in range(60001):
            # Get the data of next batch
            batch = mnist.train.next_batch(100)
            if (i % 600 == 0) and (i != 0):
                if i == 30000:
                    rt = 3e-4
                if i == 42000:
                    rt = 1e-4
                if i == 48000:
                    rt = 3e-5
                if i == 54000:
                    rt = 1e-5
                # Print the accuracy
                test_accuracy = 0
                test_accuracy_once = 0
                for index in range(200):
                    accuracy_batch = mnist.test.next_batch(50)
                    test_accuracy_once = sess.run(accuracy, feed_dict={
                        input_data: accuracy_batch[0], output_data: accuracy_batch[1],
                        phase_train: False})
                    test_accuracy += test_accuracy_once
                    test_accuracy_once = 0
                print('%g, %g, %g' %
                      (i / 600, test_accuracy / 200, (time.clock() - t0)))
                t0 = time.clock()
            # Train
            _ = sess.run(
                train_step,
                feed_dict={input_data: batch[0],
                           output_data: batch[1],
                           phase_train: True,
                           rate: rt})

if __name__ == "__main__":
    tf.app.run()