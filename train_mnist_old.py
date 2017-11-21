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


def small_cnn(x, num_conv, id, j, k, reuse, keep_prob):  #j, k not used
    print('[small_cnn] input => {}'.format(x))

    #Leaky relu
    lrelu = lambda x, alpha=0.2: tf.maximum(x, alpha * x)
    relu = lambda x: tf.nn.relu(x)

    with tf.name_scope('small_cnn'):
        with tf.variable_scope('conv1', reuse=reuse):
            W_conv1 = weight_variable_(
                [5, 5, x.get_shape().as_list()[3], 32], id, 0, 0)
            b_conv1 = bias_variable_([32], id, 0, 0)
            h_conv1 = lrelu(conv2d(x, W_conv1) + b_conv1)

        with tf.variable_scope('conv2', reuse=reuse):
            W_conv2 = weight_variable_([5, 5, 32, 64], id, 0, 0)
            b_conv2 = bias_variable_([64], id, 0, 0)
            h_conv2 = lrelu(conv2d(h_conv1, W_conv2) + b_conv2)
            h_pool1 = avg_pool(h_conv2, 2, 2)

        #'''
        with tf.variable_scope('conv3', reuse=reuse):
            W_conv3 = weight_variable_([5, 5, 64, 64], id, 0, 0)
            b_conv3 = bias_variable_([64], id, 0, 0)
            h_conv3 = lrelu(conv2d(h_pool1, W_conv3) + b_conv3)
        #'''

        with tf.variable_scope('pool2'):
            h_pool2 = avg_pool(h_conv3, 2, 2)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 64 * 16])

        with tf.variable_scope(
                'fc1',
                reuse=reuse,
                regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)):
            W_fc1 = weight_variable_([64 * 16, 1024], id, 0, 0)
            b_fc1 = bias_variable_([1024], id, 0, 0)
            h_fc1 = lrelu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.variable_scope('fc', reuse=reuse):
            W_fc = weight_variable_([1024, num_conv], id, 0, 0)
            b_fc = bias_variable_([num_conv], id, 0, 0)
            h_fc = tf.matmul(h_fc1_drop, W_fc) + b_fc

        print('[small_cnn] output <= {}'.format(h_fc))
        return h_fc
    pass


pass


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
    print('SCSCN Input: {}'.format(x))
    with tf.name_scope('kernal_size'):
        # Kernal size:
        a = 16
        b = 16

    with tf.name_scope('strides'):
        # Strides:
        stride = 3

    with tf.name_scope('pad'):
        # pad of input
        padd = 0  # no padding yet
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = tf.pad(x, [[0, 0], [padd, padd], [padd, padd], [0, 0]])
        print('SCSCN Input after padding: {}'.format(x.get_shape()))

    with tf.name_scope('input_size'):
        # x=>[-1,28,28,1]
        # Size of input:
        input_num = x.get_shape().as_list()[3]  # channels
        input_m = x.get_shape().as_list()[1]
        input_n = x.get_shape().as_list()[2]
        print('[input|SCSCN]num %d,m %d,n %d' % (input_num, input_m, input_n))

    with tf.name_scope('size'):
        # Size:
        m = int((input_m - a) / stride + 1)  #m with strides
        n = int((input_n - b) / stride + 1)

    with tf.name_scope('scn_output'):
        # Output:
        # a TensorArray of tensor used to storage the output of small_cnn
        output = tf.TensorArray('float32', num * m * n)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('fliter'):
        for h in range(num * m * n):  # h=> each conv op
            i = int(h / (m * n))  # progress of current feature map (for id)
            l = int(h % (m * n))  # ???
            j = int(l / n)  # ???/n
            k = int(l % n)  # progress at current line

            # reuse only if it's the first
            if (j == 0) and (k == 0):
                rr = False
            else:
                rr = True

            # calculate the output of the convolution fliter
            output = output.write(
                (i * m + j) * n + k,
                tf.identity(
                    small_cnn(
                        tf.slice(x, [0, j * stride, k * stride, 0],
                                 [-1, a, b, -1]), num_conv, i, j, k, rr,
                        keep_prob)))
            #small_cnn(x, num_conv, id, j, k, reuse, keep_prob) [j, k not used]
            pass
        pass

    with tf.name_scope('output_concated'):
        # return the concated and reshaped data of output
        for i in range(m):
            for j in range(n):
                for k in range(num):
                    if (j == 0) and (k == 0) and (i == 0):
                        output_ = output.read((k * m + i) * n + j)
                    else:
                        output_ = tf.concat(
                            [output_,
                             output.read((k * m + i) * n + j)], 1)

    with tf.name_scope('global_avg_pool'):
        # Global avg pooling
        scscn_out = tf.reshape(
            avg_pool(tf.reshape(output_, [-1, m, n, num * num_conv]), 5, 5),
            [-1, num_conv]), keep_prob
        print('SCSCN Output: {}'.format(scscn_out))
        return scscn_out
    pass


def weight_variable(shape):  #This fn is never used?
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

        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        cross_entropy = tf.reduce_mean(cross_entropy) #+ tf.reduce_mean(reg_losses)

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
        rt = 0.001
        train_loss = 0
        #for i in range(150000):
        for i in range(150000):
            # Get the data of next batch
            bs = 64
            batch = mnist.train.next_batch(bs)
            #if i % 1000 == 0:
            if i % 1000 == 0:
                if i == 6000:
                    rt = 3e-5
                    print('new rt: {}'.format(rt))

                # Print the accuracy
                train_accuracy = 0
                validation_loss = 0
                for index in range(50):
                    accuracy_batch = mnist.test.next_batch(200)
                    new_acc, v_loss = sess.run(
                        [accuracy, cross_entropy],
                        feed_dict={
                            x: accuracy_batch[0],
                            y_: accuracy_batch[1],
                            keep_prob: 1.0
                        })
                    train_accuracy += new_acc
                    validation_loss += v_loss
                    pass
                train_accuracy /= 50
                validation_loss /= 50

                print(
                    'epoch: %g|acc: %g|time: %g|v_loss: %g|train_loss: %g|overfit: %g|'
                    % (i, train_accuracy, (time.clock() - t0), validation_loss,
                       train_loss, validation_loss - train_loss))
                t0 = time.clock()

                # Log loss
                summary = summary_op.eval(feed_dict={
                    x: accuracy_batch[0],
                    y_: accuracy_batch[1],
                    keep_prob: 1.0
                })
                writer.add_summary(summary, i * bs)

                # Save parameters
                if (i % 5000 == 0):
                    real_location = saver.save(
                        sess, save_location, global_step=999999)
                    print("[saver] Model saved at {}".format(real_location))
                    pass
                pass

            # Train
            _, train_loss = sess.run(
                [train_step, cross_entropy],
                feed_dict={
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
