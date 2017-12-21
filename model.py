import argparse
import sys
import os
import tempfile
import time
import numpy as np
import random
import math
import logging

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from configs import *

# num_conv=10,x=[?,16,16,1]


def small_cnn(x,
              num_conv,
              keep_prob,
              id=0,
              j=0,
              k=0,
              reuse=False,
              name='small_cnn'):

    print('[small_cnn] input => {}'.format(x))
    lrelu = lambda x, alpha=0.2: tf.maximum(x, alpha * x)
    relu = lambda x: tf.nn.relu(x)
    elu = lambda x: tf.nn.elu(x)
    swish = lambda x: (x * tf.nn.sigmoid(x))
    activation = swish  # Activation Func to use

    with tf.variable_scope(name, reuse=reuse):
        # '''
        # [?,16,16,1]=>[?,12,12,32]
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=activation)
        print('[small_cnn] conv1 == {}'.format(x))

        x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=[1, 1])
        print('[small_cnn] pool1== {}'.format(x))

        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=activation)
        print('[small_cnn] conv3 == {}'.format(x))

        x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=[1, 1])
        print('[small_cnn] pool2== {}'.format(x))

        x = tf.reshape(x, [-1, 14 * 14 * 64])

        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.dense(x, 256, activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.dense(x, 128, activation=activation)
        x = tf.nn.dropout(x, keep_prob)

        x = tf.layers.dense(x, 10, activation=activation)
        pass

    print('[small_cnn] output <= {}'.format(x))
    return x


# x=>[bs,784]
# num=>num of output channels
def scscn(x, num, num_conv, e_size=1):
    phase_train = tf.placeholder(dtype=tf.bool)
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
        print('SCSCN Input after padding: {}'.format(x.get_shape()))

    with tf.name_scope('batch_normalization'):
        x = batch_norm(x, phase_train)

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
        print('m: {}\nn: {}'.format(m, n))
        print('----------------------------')

    with tf.name_scope('output'):
        # Output:
        # a TensorArray of tensor used to storage the output of small_cnn
        slicing = tf.TensorArray('float32', num * m * n)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('fliter'):
        for h in range(num * m * n):
            i = int(h / (m * n))
            l = int(h % (m * n))
            j = int(l / n)
            k = int(l % n)
            print('h: {}\ni: {}\nl: {}\nj :{}\nk: {}'.format(h, i, l, j, l))
            print('slice: {} | {}'.format([0, j * stride, k * stride, 0],
                                          [-1, a, b, -1]))
            print('----------------------------')
            slicing = slicing.write(h,
                                    tf.slice(x, [0, j * stride, k * stride, 0],
                                             [-1, a, b, -1]))
    with tf.name_scope('scn_ensemble'):
        scn_input = slicing.concat()
        print('[slicing]: {}'.format(scn_input))
        slicing.close().mark_used()

        output = small_cnn(scn_input, num_conv, keep_prob, name='scn1')
        output = tf.reshape(output, [m * n, -1, num_conv])
        output = tf.reduce_mean(output, 0)
        print('[ensemble_reshaped_output]: {}'.format(output))

        for es in range(e_size - 1):
            o = small_cnn(
                scn_input, num_conv, keep_prob, name='scn' + str(es + 2))
            o = tf.reshape(o, [m * n, -1, num_conv])
            o = tf.reduce_mean(o, 0)
            outout += o
            print('[ensemble_reshaped_output{}]: {}'.format(es + 2, output1))
            pass

        print('[ensemble_reshaped_output]: {}'.format(output))
        return output, keep_prob, phase_train


def batch_norm(x, phase_train, n_out=1):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(
            tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(
            tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(
            phase_train, mean_var_with_update,
            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
