import argparse
import sys
import os
import tempfile
import time
import numpy as np
import random
import math
import logging
import math

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from configs import *


# num_conv=10,x=[?,16,16,1]
def small_cnn(x,
              num_conv,
              keep_prob,
              phase_train,
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
    identical = lambda x: x
    activation = relu  # Activation Func to use

    with tf.variable_scope(name, reuse=reuse):

        x = conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=activation,
            strides=[1, 1],
            scope_name=1,
            use_lsuv=use_lsuv)

        x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=[2, 2])
        x = tf.nn.dropout(x, keep_prob * 1.7)
        print('[small_cnn] pool1== {}'.format(x))

        x = conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=activation,
            scope_name=2,
            strides=[1, 1],
            use_lsuv=use_lsuv)
        x = conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=activation,
            scope_name=3,
            strides=[1, 1],
            use_lsuv=use_lsuv)

        x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=[2, 2])
        print('[small_cnn] pool2== {}'.format(x))

        x = tf.reshape(
            x, [-1, x.get_shape()[1] * x.get_shape()[2] * x.get_shape()[3]])

        x = tf.nn.dropout(x, keep_prob)
        x = dense(x, 256, 1, activation=activation, use_lsuv=use_lsuv)
        x = tf.nn.dropout(x, keep_prob)
        x = dense(x, 128, 2, activation=activation, use_lsuv=use_lsuv)
        x = tf.nn.dropout(x, keep_prob)

        x = dense(x, 10, 3, activation=activation, use_lsuv=use_lsuv)
        pass

    print('[small_cnn] output <= {}'.format(x))
    return x


# x=>[bs,784]
# num=>num of output channels
def scscn(x, num, num_conv, e_size=1, keep_prob=None, phase_train=None):
    with tf.name_scope('kernal_size'):
        # Kernal size:
        a = 16
        b = 16

    with tf.name_scope('strides'):
        # Strides:
        stride = 4

    with tf.name_scope('pad'):
        # pad of input
        padd = 0
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = tf.pad(x, [[0, 0], [padd, padd], [padd, padd], [0, 0]])
        print('SCSCN Input after padding: {}'.format(x.get_shape()))

    with tf.name_scope('batch_normalization'):
        x = batch_norm(x, phase_train)
        pass

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

    # with tf.name_scope('dropout'):

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

        variance_cal_scnn = []
        mse_scnn = 0
        output = None

        for es in range(e_size):
            with tf.variable_scope('scn' + str(es)):
                print('es{}--------------------------'.format(es))
                o = small_cnn(
                    scn_input,
                    num_conv,
                    keep_prob,
                    phase_train,
                    name='scn' + str(es))
                o = tf.reshape(o, [m * n, -1, num_conv])
                o = tf.reduce_mean(o, 0)
                variance_cal_scnn.append(o)
                if output == None:
                    output = o
                else:
                    output += o
                print('[ensemble_reshaped_output{}]: {}'.format(
                    es + 1, output))
                print('es{}--------------------------'.format(es))
                pass
        print('[ensemble_reshaped_output_all]: {}'.format(output))

        for o in variance_cal_scnn:
            for o_ in variance_cal_scnn:
                # TODO: Add variance cal
                mse_scnn += tf.losses.mean_squared_error(o, o_)
                pass
            values_to_log.append(
                tf.summary.image(o.name, tf.reshape(o, [-1, 2, 5, 1])))
            pass
        values_to_log.append(tf.summary.scalar("mse", mse))

        return output, phase_train


def depthwise_conv2d(x, W):
    # print('[dwc]: {}\n{}'.format(x, W))
    return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def lsuv(layer, w):
    print('[LSUV]: Treating weight {} at {}'.format(layer, w))
    batch = mnist.test.next_batch(bs)
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        # global tol,t_max
        blob, new_weight = sess.run(
            [layer, w],
            feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: 1,
                phase_train: False
            })

    t = 0
    var = np.var(blob)
    while (np.abs(var - 1) > tol and t < t_max):
        print('[LSUV]: Variance larger than tolerance: {} > {}'.format(
            np.abs(var - 1), tol))
        old_weight = new_weight
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            blob = sess.run(
                layer,
                feed_dict={
                    x: batch[0],
                    y_: batch[1],
                    keep_prob: 1,
                    phase_train: False
                })
        t += 1
        var = np.var(blob)
        new_weight = old_weight / (var**0.5)
        # print('Old: {}\nNew: {}'.format(old_weight, new_weight))
        w.assign(new_weight)
        pass
    print('[LSUV]: {} in {} treated over {} iterations'.format(w, layer, t))
    return layer


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.054)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def weight_variable_(shape, id, j, k, initializer=tf.orthogonal_initializer()):
    # initializer=tf.random_normal_initializer(0, 0.1)):
    print("weights" + str(id) + "a" + str(j) + "a" + str(k))
    return tf.get_variable("weights" + str(id) + "a" + str(j) + "a" + str(k),
                           shape, None, initializer)


def bias_variable_(shape, id, j, k, initializer=tf.constant_initializer(0)):
    return tf.get_variable("biases" + str(id) + "a" + str(j) + "a" + str(k),
                           shape, None, initializer)


def depthwise_conv2d(inputs, W, scope_name=None):
    return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def conv2d(inputs,
           filters,
           kernel_size,
           scope_name=None,
           padding='SAME',
           activation=tf.nn.relu,
           strides=[1, 1],
           id=0,
           use_lsuv=False,
           use_bn=use_bn,
           reuse=False,
           draw=False):
    x = inputs
    strides = [1, strides[0], strides[1], 1]
    scope_name = 'conv' + str(scope_name)
    padding = padding.upper()

    with tf.variable_scope(scope_name, reuse=reuse):
        if not use_lsuv:
            w = weight_variable_(
                [kernel_size[0], kernel_size[1],
                 x.get_shape()[3], filters], id, 0, 0)
            b = bias_variable_([filters], id, 0, 0)
            x = tf.nn.conv2d(x, w, strides=strides, padding=padding) + b
            # if use_bn:
            # x = batch_norm(x, phase_train, n_out=x.get_shape()[3])
            x = activation(x)
        else:
            w = weight_variable_(
                [kernel_size[0], kernel_size[1],
                 x.get_shape()[3], filters],
                id,
                0,
                0,
                initializer=tf.initializers.orthogonal())
            b = bias_variable_([filters], id, 0, 0)
            x = tf.nn.conv2d(x, w, strides=strides, padding=padding) + b
            x = activation(x)
            x = lsuv(x, w)

        ws = tf.unstack(w, axis=3)
        # for w in ws:
        # values_to_log.append(
        # tf.summary.image(
        # w.name,
        # tf.reshape(w, [-1, kernel_size[0], kernel_size[1], 1])))
        # print('[small_cnn] conv' + scope_name + ' == {}'.format(x))
        return x


def dense(x,
          num,
          scope_name,
          id=0,
          use_lsuv=False,
          activation=tf.nn.relu,
          reuse=False):
    with tf.variable_scope('fc' + str(scope_name), reuse=reuse):
        w = weight_variable_([x.get_shape()[1], num], id, 0, 0)
        b = bias_variable_([num], id, 0, 0)
        x = activation(tf.matmul(x, w) + b)
        if not use_lsuv:
            return x
        else:
            return lsuv(x, w)


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
    if use_bn:
        return normed
    else:
        return x


# Placehoder of input and output
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='input')

    y_ = tf.placeholder(tf.float32, [None, 10], name='validation')

    # Necessary to be kept visible
    keep_prob = tf.placeholder(tf.float32)
    global phase_train
    phase_train = tf.placeholder(dtype=tf.bool)
    # The main model
    y_conv, phase_train = scscn(
        x,
        num=1,
        num_conv=10,
        e_size=e_size,
        keep_prob=keep_prob,
        phase_train=phase_train)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_, logits=y_conv)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    cross_entropy = tf.reduce_mean(
        cross_entropy)  #+ tf.reduce_mean(reg_losses)

with tf.name_scope('adam_optimizer'):
    rate = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)
#"""
with tf.name_scope('momentum_optimizer'):  #this works really bad...
    train_step_mmntm = tf.train.MomentumOptimizer(
        rate, momentum=0.9).minimize(cross_entropy)
#"""

with tf.name_scope('accuracy'):
    print('[y_conv]: {}'.format(y_conv))
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

with tf.name_scope('logger'):
    # Graph
    print('Saving graph to: %s' % graph_location)
    writer = tf.summary.FileWriter(
        graph_location, graph=tf.get_default_graph())
    saver = tf.train.Saver()
    # Loss
    tf.summary.scalar("t_loss", cross_entropy)
    tf.summary.scalar("t_acc", accuracy)
    summary_op = tf.summary.merge_all()
