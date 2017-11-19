from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import math

FLAGS = None

# Small CNN:
# convolution fliter of SCSCN


def small_cnn(x, num_conv, keep_prob, id = 0, j = 0, k = 0, reuse = False):
    with tf.variable_scope('conv1', reuse = reuse):
        W_conv1 = weight_variable_(
            [5,
             5,
             x.get_shape().as_list()[3],
             32],
            id, 0, 0)
        b_conv1 = bias_variable_([32], id, 0, 0)
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    with tf.variable_scope('conv2', reuse = reuse):
        W_conv2 = weight_variable_(
            [5,
             5,
             32,
             64],
            id, 0, 0)
        b_conv2 = bias_variable_([64], id, 0, 0)
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        h_pool1 = avg_pool(h_conv2, 2, 2)

    with tf.variable_scope('conv3', reuse = reuse):
        W_conv3 = weight_variable_(
            [5,
             5,
             64,
             64],
            id, 0, 0)
        b_conv3 = bias_variable_([64], id, 0, 0)
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    with tf.variable_scope('pool2'):
        h_pool2 = avg_pool(h_conv3, 2, 2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 64 * 16])

    with tf.variable_scope('fc1', reuse = reuse):
        W_fc1 = weight_variable_([64 * 16, 1024], id, 0, 0)
        b_fc1 = bias_variable_([1024], id, 0, 0)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('fc', reuse = reuse):
        W_fc = weight_variable_([1024, num_conv], id, 0, 0)
        b_fc = bias_variable_([num_conv], id, 0, 0)
        h_fc = tf.matmul(h_fc1_drop, W_fc) + b_fc

    return h_fc


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def avg_pool(x, m, n):
    return tf.nn.avg_pool(x, ksize=[1, m, n, 1],
                          strides=[1, m, n, 1], padding='SAME')

def max_pool(x, m, n):
    return tf.nn.max_pool(x, ksize=[1, m, n, 1],
                          strides=[1, m, n, 1], padding='SAME')

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
        slicing = tf.TensorArray('float32', num * m * n)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('fliter'):
        for h in range(num * m * n):
            i = int(h / (m*n))
            l = int(h % (m*n))
            j = int(l / n)
            k = int(l % n)
            slicing = slicing.write(h,
                        tf.slice(x, [0, j * stride, k * stride, 0],
                                    [-1, a, b, -1]))
    with tf.name_scope('scn'):
        scn_input = slicing.concat()
        output = small_cnn(scn_input, num_conv, keep_prob)
        output = tf.reshape(output, [m * n, -1, num * num_conv])

    return tf.reduce_mean(output, 0), keep_prob


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
    # Read data from MNIST
    # mnist = input_data.read_data_sets('fashion', one_hot=True)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Placehoder of input and output
    x = tf.placeholder(tf.float32, [None, 784])

    y_ = tf.placeholder(tf.float32, [None, 10])

    # The main model
    y_conv, keep_prob = scscn(x, 1, 10)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        rate = tf.placeholder(tf.float32)
        train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.name_scope('config'):
        config = tf.ConfigProto(
            inter_op_parallelism_threads = 256,
            intra_op_parallelism_threads = 64
        )
        config.gpu_options.allow_growth = True

    with tf.name_scope('graph'):
        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())
        merged = tf.summary.merge_all()

    # Start to run
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        t0 = time.clock()
        rt = 1e-3
        for i in range(140001):
            # Get the data of next batch
            batch = mnist.train.next_batch(5)
            if i % 1000 == 0:
                # rt = rt * 0.95
                if i == 600000:
                    rt = 3e-4
                if i == 1000000:
                    rt = 1e-4
                # Print the accuracy
                train_accuracy = 0
                for index in range(50):
                    accuracy_batch = mnist.test.next_batch(200)
                    train_accuracy += accuracy.eval(feed_dict={
                        x: accuracy_batch[0],
                            y_: accuracy_batch[1], keep_prob: 1.0})
                print(
                    '%g, %g, %g' %
                    (i / 1000, train_accuracy / 50, (time.clock() - t0)))
                t0 = time.clock()
            # Train
            train_step.run(
                feed_dict={x: batch[0],
                           y_: batch[1],
                           keep_prob: 0.5,
                           rate: rt})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
