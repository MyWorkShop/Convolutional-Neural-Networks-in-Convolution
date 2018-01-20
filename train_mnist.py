from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time
import datetime

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

import tensorflow as tf

FLAGS = None

# Small CNN:
# convolution fliter of SCSCN


def small_cnn(x, num_conv, keep_prob, id=0, j=0, k=0, reuse=False):
    swish = lambda x: (x * tf.nn.sigmoid(x))
    activation = tf.nn.relu  # Activation Func to use
    with tf.variable_scope('conv1', reuse=reuse):
        W_conv1 = weight_variable(
            [5,
             5,
             x.get_shape().as_list()[3],
             16])
        b_conv1 = bias_variable([16])
        h_conv1 = activation(conv2d(x, W_conv1) + b_conv1)
        summary_layer(W_conv1, b_conv1)
        summary_output(h_conv1)

    with tf.variable_scope('conv2', reuse=reuse):
        W_conv2 = weight_variable([5, 5, 16, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        summary_layer(W_conv2, b_conv2)
        summary_output(h_conv2)
        h_pool2 = tf.nn.dropout(avg_pool(h_conv2, 2, 2), keep_prob)

    with tf.variable_scope('conv3', reuse=reuse):
        W_conv3 = weight_variable([5, 5, 32, 32])
        b_conv3 = bias_variable([32])
        h_conv3 = activation(conv2d(h_pool2, W_conv3) + b_conv3)
        summary_layer(W_conv3, b_conv3)
        summary_output(h_conv3)

    with tf.variable_scope('pool2'):
        h_pool2 = tf.nn.dropout(avg_pool(h_conv3, 2, 2), keep_prob)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 16])

    with tf.variable_scope('fc1', reuse=False):
        W_fc1 = weight_variable([32 * 16, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = activation(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        summary_layer(W_fc1, b_fc1)

    with tf.variable_scope('fc', reuse=reuse):
        W_fc = weight_variable([1024, num_conv])
        b_fc = bias_variable([num_conv])
        h_fc = tf.matmul(h_fc1_drop, W_fc) + b_fc
        summary_layer(W_fc, b_fc)

    return tf.nn.dropout(h_fc, keep_prob)


def summary_layer(weight, bias):
    with tf.name_scope('layer_summaries'):
        with tf.name_scope('weight'):
            variable_summaries(weight)
        with tf.name_scope('bias'):
            variable_summaries(bias)


def summary_output(output):
    with tf.name_scope('summaries_output'):
        output = tf.transpose(output, [0, 3, 1, 2])
        output = tf.reshape(
            output, [-1, output.get_shape().as_list()[2], output.get_shape().as_list()[3], 1])
        tf.summary.image('output', output, 10)


def variable_summaries(var):
    with tf.name_scope('summaries_var'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


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

    with tf.name_scope('input'):
        summary_output(x)

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
            i = int(h / (m * n))
            l = int(h % (m * n))
            j = int(l / n)
            k = int(l % n)
            slicing = slicing.write(
                h, tf.slice(x, [0, j * stride, k * stride, 0],
                            [-1, a, b, -1]))
        scn_input = slicing.concat()
        slicing.close().mark_used()
    with tf.name_scope('samll_cnn'):
        scn = small_cnn(scn_input, num_conv, keep_prob)
        scn = tf.reshape(scn, [m, n, -1, num_conv])
        scn = tf.transpose(scn, [2, 3, 0, 1])
        draw = tf.reshape(scn, [-1, m, n, 1])
        summary_output(draw)
    # with tf.name_scope('depthwiseconv'):
    #     x_dwc = tf.transpose(scn, [0, 2, 3, 1])
    #     W_dwc = weight_variable([m, n, num_conv, 1])
    #     h_dwc = depthwise_conv2d(x_dwc, W_dwc)
    # with tf.name_scope('output'):
    #     output = tf.reshape(h_dwc, [-1, num_conv])
    with tf.name_scope('output'):
        output = tf.reduce_mean(scn, [2, 3])

    return output, keep_prob


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


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
        train_step = tf.train.AdamOptimizer(
            rate).minimize(tf.sqrt(cross_entropy))

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.name_scope('config'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    with tf.name_scope('tensorboard'):
        graph_location = './log/' + str(datetime.datetime.now())
        test_location = './log_test/' + str(datetime.datetime.now())
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        test_writer = tf.summary.FileWriter(test_location)
        train_writer.add_graph(tf.get_default_graph())
        tf.summary.scalar('loss', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()

    # Start to run
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        t0 = time.clock()
        rt = 1e-3
        train_loss = 0
        for i in range(60001):
            # Get the data of next batch
            batch = mnist.train.next_batch(100)
            if (i % 600 == 0) and (i != 0):
                if i == 30000:
                    rt = 3e-4
                if i == 42000:
                    rt = 1e-4
                # Print the accuracy
                test_accuracy = 0
                test_accuracy_once = 0
                for index in range(200):
                    accuracy_batch = mnist.test.next_batch(50)
                    summary, test_accuracy_once = sess.run([merged, accuracy], feed_dict={
                        x: accuracy_batch[0],
                            y_: accuracy_batch[1], keep_prob: 1.0})
                    test_accuracy += test_accuracy_once
                    test_accuracy_once = 0
                print('%g, %g, %g' %
                      (i / 600, test_accuracy / 200, (time.clock() - t0)))
                test_summary = tf.Summary()
                test_summary.value.add(
                    tag="accuracy", simple_value=test_accuracy / 200)
                test_writer.add_summary(test_summary, i / 600)
                t0 = time.clock()
                train_loss = 0
            # Train
            train_loss_once, summary, _ = sess.run(
                [cross_entropy, merged, train_step],
                                                   feed_dict={x: batch[0],
                                                              y_: batch[1],
                                                              keep_prob: 0.5,
                                                              rate: rt})
            train_loss += train_loss_once
            train_loss_once = 0
            train_writer.add_summary(summary, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
