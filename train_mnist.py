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

# TODO:
# new activation fn

# Small CNN:
# convolution fliter of SCSCN


# TODO: argument dismatch
#def small_cnn(x, num_conv, id, j, k, reuse, keep_prob):
def small_cnn(x, num_conv, keep_prob, id=0, j=0, k=0, reuse=False):
    print('[small_cnn] input => {}'.format(x))
    lrelu = lambda x, alpha=0.2: tf.maximum(x, alpha * x)
    relu = lambda x: tf.nn.relu(x)
    elu = lambda x: tf.nn.elu(x)
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
        W_conv3 = weight_variable_([3, 3, 64, 64], id, 0, 0)
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

    print('[small_cnn] output <= {}'.format(h_fc))
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


# x=>[bs,784]
# num=>num of output channels
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
        print('SCSCN Input after padding: {}'.format(x.get_shape()))

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
    with tf.name_scope('scn'):
        scn_input = slicing.concat()
        output = small_cnn(scn_input, num_conv, keep_prob)
        output = tf.reshape(output, [m * n, -1, num * num_conv])

    return tf.reduce_mean(output, 0), keep_prob


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):  #This fn is never used?
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


#Reuse if exists
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
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='input')

        y_ = tf.placeholder(tf.float32, [None, 10], name='validation')

    # The main model
    y_conv, keep_prob = scscn(x, 1, 10)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y_conv)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        cross_entropy = tf.reduce_mean(cross_entropy) + tf.reduce_mean(
            reg_losses)

    with tf.name_scope('adam_optimizer'):
        rate = tf.placeholder(tf.float32)
        train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)
    #"""
    with tf.name_scope('momentum_optimizer'):  #this works really bad...
        train_step_mmntm = tf.train.MomentumOptimizer(
            rate, momentum=0.9).minimize(cross_entropy)
    #"""

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.name_scope('config'):
        config = tf.ConfigProto(
            inter_op_parallelism_threads=256, intra_op_parallelism_threads=64)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.name_scope('logger'):
        # Graph
        run_description = 'l2_lrelu'
        import time

        graph_location = '/tmp/saved_models/' + run_description  #+ str(time.time())
        print('Saving graph to: %s' % graph_location)

        writer = tf.summary.FileWriter(
            graph_location, graph=tf.get_default_graph())

        saver = tf.train.Saver()
        save_location = '/tmp/saved_models/' + run_description + '/saved'
        recover_location = '/tmp/saved_models/' + run_description + '/'

        # Loss
        tf.summary.scalar("loss", cross_entropy)
        summary_op = tf.summary.merge_all()

    #"""
    # Start to run
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        try:
            import os
            if (True):
                saver.restore(sess,
                              tf.train.latest_checkpoint(recover_location))
                print('[saver] Parameter loaded from {}'.format(
                    tf.train.latest_checkpoint(recover_location)))
            else:
                print('[saver] Checkpoint not found.')
        except Exception as e:
            print('[saver] Failed to load parameter: {}'.format(e))

        t0 = time.clock()

        rt = 1e-3
        for i in range(140001):
            # Get the data of next batch
            batch = mnist.train.next_batch(60)
            if i % 1000 == 0:
                # rt = rt * 0.95
                if i == 600000:
                    rt = 3e-4
                    print('new rt: {}'.format(rt))
                if i == 1000000:
                    rt = 1e-4
                    print('new rt: {}'.format(rt))

                # Print the accuracy
                train_accuracy = 0
                for index in range(50):
                    accuracy_batch = mnist.test.next_batch(200)
                    train_accuracy += accuracy.eval(feed_dict={
                        x: accuracy_batch[0],
                        y_: accuracy_batch[1],
                        keep_prob: 1.0
                    })

                print('%g, %g, %g' % (i, train_accuracy / 50,
                                      (time.clock() - t0)))
                t0 = time.clock()

                # Log loss
                summary = summary_op.eval(feed_dict={
                    x: accuracy_batch[0],
                    y_: accuracy_batch[1],
                    keep_prob: 1.0
                })
                writer.add_summary(summary, i * bs)
                # Save parameters
                if (i % 1000 == 0):
                    real_location = saver.save(
                        sess, save_location, global_step=999999
                    )  # Making sure it's the lastest_checkpoint
                    print("[saver] Model saved at {}".format(real_location))
                    pass
                pass

            # Train
            train_step.run(feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: 0.5,
                rate: rt
            })
    #"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
