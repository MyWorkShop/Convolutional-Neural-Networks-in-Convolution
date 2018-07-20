from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import time


def wrap(x, m, n, stride, shape):
    slicing = tf.TensorArray('float32', m * n)
    for j in range(m):
        for k in range(n):
            slicing = slicing.write(j * n + k,
                                    tf.slice(x, [0, j * stride, k * stride, 0],
                                             shape))
    sliced = tf.reshape(slicing.concat(), shape)
    slicing.close().mark_used()
    return sliced


def unwrap(x, m, n, stride, shape):
    adding = tf.TensorArray('float32', m * n)
    to_split = tf.reshape(x, tf.concat([[m * n], shape], 0))
    split = tf.TensorArray('float32', m * n).unstack(to_split)
    for j in range(m):
        for k in range(n):
            adding = adding.write(
                j * n + k,
                tf.pad(
                    split.read(j * n + k),
                    [[0, 0], [j * stride, (m - 1 - j) * stride],
                     [k * stride, (n - 1 - k) * stride], [0, 0]]))
    split.close().mark_used()
    added = tf.reduce_sum(adding.stack(), 0)
    adding.close().mark_used()
    return added


def fcn(x, phase_train):
    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)
    conv1_dropout = tf.layers.dropout(
        inputs=conv1, rate=RATE_DROPOUT, training=phase_train)

    # Convolutional layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1_dropout,
        filters=32,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)
    return conv2


def coord_layer(x):
    size = [x.get_shape().as_list()[1], x.get_shape().as_list()[2]]
    layer1 = []
    layer2 = []
    for i in range(size[0]):
        layer1.append(
            tf.constant(
                value=(np.ones([size[1]]) * i - size[0] / 2.) / size[0] * 2.,
                shape=[size[1]],
                dtype=tf.float32))
    for i in range(size[1]):
        layer2.append(
            tf.constant(
                value=(np.ones([size[0]]) * i - size[1] / 2.) / size[1] * 2.,
                shape=[size[0]],
                dtype=tf.float32))
    layer1 = tf.stack(layer1, 0)
    layer1 = tf.reshape(layer1, size)
    layer2 = tf.transpose(tf.reshape(tf.concat(layer2, 0), np.transpose(size)))
    layer = tf.expand_dims(tf.stack([layer1, layer2], axis=2), 0)
    layer = tf.tile(layer, [tf.shape(x)[0], 1, 1, 1])
    xstack = tf.unstack(x, axis=3)
    lstack = tf.unstack(layer, axis=3)
    x = tf.stack(xstack + lstack, axis=3)
    print(x)
    return x


def small_cnn(x, phase_train):
    # Convolutional Layer #1
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    x = coord_layer(x)
    # Convolutional Layer #2
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    x = coord_layer(x)
    # Pooling Layer #1
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 #4 and Pooling Layer #2
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    x = coord_layer(x)
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    # Dense Layer
    x = tf.layers.dropout(
        inputs=tf.reshape(x, [-1, 4 * 4 * 32]),
        rate=0.,
        training=phase_train)
    x = tf.layers.dense(
        inputs=x, units=1024, activation=tf.nn.relu)
    x = tf.layers.dropout(inputs=x, rate=0.4, training=phase_train)

    # Logits Layer
    x = tf.layers.dense(inputs=x, units=10)

    return x


def cnnic(x):
    phase_train = tf.placeholder(tf.bool)

    m = 5
    n = 5
    stride = 3
    x = tf.reshape(x, [-1, 28, 28, 1])
    x = coord_layer(x)
    # cords = tf.cond(phase_train, lambda: tf.tile(x, multiples=[100, 1, 1, 1]),
    # lambda: tf.tile(x, multiples=[50, 1, 1, 1]))
    print('cords: {}'.format(x))

    #Input of CNNIC
    slicing = tf.TensorArray('float32', m * n)
    for j in range(m):
        for k in range(n):
            slicing = slicing.write(j * n + k,
                                    tf.slice(x, [0, j * stride, k * stride, 0],
                                             [-1, 16, 16, 3]))
    scn_input = tf.reshape(slicing.concat(), [-1, 16, 16, 3])
    slicing.close().mark_used()

    scn_output_raw = small_cnn(scn_input, phase_train)
    scn_output = tf.reshape(scn_output_raw, [m * n, -1, 10])
    cnnic_output = tf.reduce_mean(scn_output, 0)

    return cnnic_output, phase_train


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    input_data = tf.placeholder(tf.float32, [None, 784])
    output_data = tf.placeholder(tf.int64, [None])
    y_model, phase_train = cnnic(input_data)

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

    with tf.Session() as sess:
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
                    test_accuracy_once = sess.run(
                        accuracy,
                        feed_dict={
                            input_data: accuracy_batch[0],
                            output_data: accuracy_batch[1],
                            phase_train: False
                        })
                    test_accuracy += test_accuracy_once
                    test_accuracy_once = 0
                print('%g, %g, %g' % (i / 600, test_accuracy / 200,
                                      (time.clock() - t0)))
                t0 = time.clock()
            # Train
            _ = sess.run(
                train_step,
                feed_dict={
                    input_data: batch[0],
                    output_data: batch[1],
                    phase_train: True,
                    rate: rt
                })


if __name__ == "__main__":
    tf.app.run()
