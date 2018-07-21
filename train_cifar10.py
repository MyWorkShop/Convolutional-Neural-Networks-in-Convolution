from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import time

import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
import math


def get_data_set(name="train"):
    x = None
    y = None

    maybe_download_and_extract()

    folder_name = "cifar_10"

    f = open('./data_set/' + folder_name + '/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open('./data_set/' + folder_name + '/data_batch_' + str(i + 1),
                     'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32 * 32 * 3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/' + folder_name + '/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32 * 32 * 3)

    return x, (y)


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory + "cifar_10/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(
            url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(
                file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(
                name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory + "./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)


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
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Convolutional Layer #2
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    x = tf.layers.dropout(inputs=x, rate=0.4, training=phase_train)
    # Pooling Layer #1
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 #4 and Pooling Layer #2
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    x = tf.layers.dropout(inputs=x, rate=0.4, training=phase_train)
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    # Dense Layer
    x = tf.layers.dropout(
        inputs=tf.reshape(x, [-1, 4 * 4 * 32]), rate=0.4, training=phase_train)
    # x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu)
    # x = tf.layers.dropout(inputs=x, rate=0.4, training=phase_train)
    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu)
    x = tf.layers.dropout(inputs=x, rate=0.4, training=phase_train)
    x = tf.layers.dense(inputs=x, units=32, activation=tf.nn.relu)
    # Logits Layer
    # x = tf.layers.dense(inputs=x, units=10)
    return x

    return x


def cnnic(x):
    phase_train = tf.placeholder(tf.bool)

    m = 5
    n = 5
    stride = 4
    x = tf.reshape(x, [-1, 32, 32, 3])
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
    # scn_output = tf.reshape(scn_output_raw, [m * n, -1, 10])
    # cnnic_output = tf.reduce_mean(scn_output, 0)
    scn_output = tf.reshape(scn_output_raw, [m * n, -1, 32])
    scn_output = tf.transpose(scn_output, [1, 0, 2])
    scn_output = tf.reshape(scn_output, [-1, m * n * 32])
    scn_output = tf.layers.dropout(
        inputs=scn_output, rate=0.4, training=phase_train)
    scn_output = tf.layers.dense(
        tf.reshape(scn_output, [-1, m * n * 32]), 1024)
    scn_output = tf.layers.dropout(
        inputs=scn_output, rate=0.4, training=phase_train)
    cnnic_output = tf.layers.dense(scn_output, 10)

    return cnnic_output, phase_train


def main(unused_argv):

    input_data = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
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
        rt = 5e-4
        ITERATION_SIZE = 30
        BATCH_SIZE = 256
        train_x, train_y = get_data_set("train")
        test_x, test_y = get_data_set("test")
        batch_count = int(math.ceil(len(train_x) / BATCH_SIZE))
        test_count = int(math.ceil(len(test_x) / BATCH_SIZE))
        print('\tbatch count: {} \n\ttest count: {}'.format(
            batch_count, test_count))
        for i in range(ITERATION_SIZE):
            for c in range(batch_count):
                batch_xs = train_x[c * BATCH_SIZE:(c + 1) * BATCH_SIZE]
                batch_ys = train_y[c * BATCH_SIZE:(c + 1) * BATCH_SIZE]
                if (c + 1) * (i + 1) % 300 == 0:
                    # Print the accuracy
                    test_accuracy = 0
                    test_accuracy_once = 0
                    for index in range(test_count):
                        test_xs = test_x[index * BATCH_SIZE:(
                            index + 1) * BATCH_SIZE]
                        test_ys = test_y[index * BATCH_SIZE:(
                            index + 1) * BATCH_SIZE]
                        test_accuracy_once = sess.run(
                            accuracy,
                            feed_dict={
                                input_data: test_xs,
                                output_data: test_ys,
                                phase_train: False
                            })
                        test_accuracy += test_accuracy_once
                        test_accuracy_once = 0
                    print('%g, %g, %g' %
                          ((i + 1) * (c + 1), test_accuracy / test_count,
                           (time.clock() - t0)))
                    t0 = time.clock()
                # Train
                _ = sess.run(
                    train_step,
                    feed_dict={
                        input_data: batch_xs,
                        output_data: batch_ys,
                        phase_train: True,
                        rate: rt
                    })


if __name__ == "__main__":
    tf.app.run()
