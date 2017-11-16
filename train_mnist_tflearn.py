#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time

from tensorflow.examples.tutorials.mnist import input_data
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
from tflearn.datasets import mnist

import tensorflow as tf

import math

FLAGS = None
tf.logging.set_verbosity(tf.logging.INFO)

# Small CNN:
# convolution fliter of SCSCN
#TODO/NOTE:
#1. tflearn version of implementation. --hxb
#2. default log dir: "/tmp/tflearn_logs". --hxb
#3. WTF is keep_prob? --hxb


def small_cnn(x, num_conv, id, j, k, reuse, keep_prob):
    with tf.variable_scope('conv1', reuse=reuse):
        W_conv1 = weight_variable_([5, 5, x.get_shape().as_list()[3], 32], id,
                                   0, 0)
        b_conv1 = bias_variable_([32], id, 0, 0)
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    with tf.variable_scope('conv2', reuse=reuse):
        W_conv2 = weight_variable_([5, 5, 32, 64], id, 0, 0)
        b_conv2 = bias_variable_([64], id, 0, 0)
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        h_pool1 = avg_pool(h_conv2, 2, 2)

    with tf.variable_scope('conv3', reuse=reuse):
        W_conv3 = weight_variable_([5, 5, 64, 64], id, 0, 0)
        b_conv3 = bias_variable_([64], id, 0, 0)
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    with tf.variable_scope('pool2'):
        h_pool2 = avg_pool(h_conv3, 2, 2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 64 * 16])

    with tf.variable_scope('fc1', reuse=reuse):
        W_fc1 = weight_variable_([64 * 16, 1024], id, 0, 0)
        b_fc1 = bias_variable_([1024], id, 0, 0)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('fc', reuse=reuse):
        W_fc = weight_variable_([1024, num_conv], id, 0, 0)
        b_fc = bias_variable_([num_conv], id, 0, 0)
        h_fc = tf.matmul(h_fc1_drop, W_fc) + b_fc

    return h_fc


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def avg_pool(x, m, n):
    return tf.nn.avg_pool(
        x, ksize=[1, m, n, 1], strides=[1, m, n, 1], padding='SAME')


def max_pool(x, m, n):
    return tf.nn.max_pool(
        x, ksize=[1, m, n, 1], strides=[1, m, n, 1], padding='SAME')


def scscn(x, num, num_conv):
    with tf.name_scope('kernal_size'):
        # Kernal size:
        a = 16
        b = 16

    with tf.name_scope('strides'):
        # Strides:
        stride = 3

    with tf.name_scope('pad'):
        # pad of input
        padd = 0
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = tf.pad(x, [[0, 0], [padd, padd], [padd, padd], [0, 0]])

    with tf.name_scope('input_size'):
        # Size of input:
        input_num = x.get_shape().as_list()[3]
        input_m = x.get_shape().as_list()[1]
        input_n = x.get_shape().as_list()[2]
        print('[input|SCSCN]num %d,m %d,n %d' % (input_num, input_m, input_n))

    with tf.name_scope('size'):
        # Size:
        m = int((input_m - a) / stride + 1)
        n = int((input_n - b) / stride + 1)

    with tf.name_scope('output'):
        # Output:
        # a TensorArray of tensor used to storage the output of small_cnn
        output = tf.TensorArray('float32', num * m * n)

    with tf.name_scope('dropout'):
        keep_prob = tf.constant(
            0.5)  # If it's not changing, why using a placeholder? --hxb
        #keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('fliter'):
        for h in range(num * m * n):
            i = int(h / (m * n))
            l = int(h % (m * n))
            j = int(l / n)
            k = int(l % n)
            if (j == 0) and (k == 0):
                rr = False
            # calculate the output of the convolution fliter
            output = output.write(
                (i * m + j) * n + k,
                tf.identity(
                    small_cnn(
                        tf.slice(x, [0, j * stride, k * stride, 0],
                                 [-1, a, b, -1]), num_conv, i, j, k, rr,
                        keep_prob)))
            rr = True
    # return the concated and reshaped data of output
    for i in range(m):
        for j in range(n):
            for k in range(num):
                if (j == 0) and (k == 0) and (i == 0):
                    output_ = output.read((k * m + i) * n + j)
                else:
                    output_ = tf.concat(
                        [output_, output.read((k * m + i) * n + j)], 1)
    return tf.reshape(
        avg_pool(tf.reshape(output_, [-1, m, n, num * num_conv]), 5, 5),
        [-1, num_conv]), keep_prob


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def weight_variable_(shape, id, j, k):
    return tf.get_variable("weights" + str(id) + "a" + str(j) + "a" + str(k),
                           shape, None, tf.random_normal_initializer(0, 0.05))


def bias_variable_(shape, id, j, k):
    return tf.get_variable("biases" + str(id) + "a" + str(j) + "a" + str(k),
                           shape, None, tf.constant_initializer(0.0))


def main(_):
    # Read data from MNIST(from tflearn)
    # mnist = input_data.read_data_sets('fashion', one_hot=True)
    #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X, Y, X_test, Y_test = mnist.load_data(one_hot=True)

    #Dummies
    img_prep = ImagePreprocessing()
    img_aug = ImageAugmentation()

    x = input_data(
        shape=[None, 784],
        data_preprocessing=img_prep,
        data_augmentation=img_aug)

    # The main model

    y_conv, keep_prob = scscn(x, 1, 10)
    #rate = tf.placeholder(tf.float32)
    rate = tf.constant(0.001)
    output = regression(
                y_conv, optimizer='adam',
                loss=lambda y_pred, y_true: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_true, logits=y_pred)),
                learning_rate=rate)

    scnscn_model = tflearn.DNN(output, tensorboard_verbose=0)
    scnscn_model.fit(
        X,
        Y,
        n_epoch=50,
        shuffle=False,
        validation_set=(X_test, Y_test),
        show_metric=True,
        batch_size=64,
        run_id='mnist' + str(time.clock()))
    """
    # Start to run
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        t0 = time.clock()
        lr = 0.001
        for i in range(150000):
            # Get the data of next batch
            batch = mnist.train.next_batch(60)
            if i % 1000 == 0:
                if i == 60000:
                    lr = 1e-4
                if i == 100000:
                    lr = 3e-5
                # Print the accuracy
                t0 = time.clock()
                train_accuracy = 0
                for index in range(50):
                    accuracy_batch = mnist.test.next_batch(200)
                    train_accuracy += accuracy.eval(feed_dict={
                        x: accuracy_batch[0],
                        y_: accuracy_batch[1],
                        keep_prob: 1.0
                    })
                train_accuracy /= 50
                print('epoch: {} |acc: {} |time: {}'.format(
                    i / 1000, train_accuracy, (time.clock() - t0)))
            # Train
            train_step.run(feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: 0.5,
                rate: lr
            })
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
