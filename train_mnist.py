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
    with tf.variable_scope(name, reuse=reuse):

        # [?,16,16,1]=>[?,12,12,32]
        x = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="valid",
            activation=tf.nn.relu)
        print('[small_cnn] conv1 == {}'.format(x))

        # [?,12,12,32]=>[?,8,8,64]
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[5, 5],
            padding="valid",
            activation=tf.nn.relu)
        print('[small_cnn] conv2 == {}'.format(x))

        # x = tf.layers.average_pooling2d(x, pool_size=[2, 2], strides=[1, 1])
        # print('[small_cnn] pool1  == {}'.format(x))

        x = tf.reshape(x, [-1, 8 * 8 * 64])

        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.dense(x, 10, activation=tf.nn.relu)
        pass

    print('[small_cnn] output <= {}'.format(x))
    return x


# x=>[bs,784]
# num=>num of output channels
def scscn(x, num, num_conv, e_size=1):
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
    with tf.name_scope('scn_ensemble'):
        scn_input = slicing.concat()
        print('[sliced_img]: {}'.format(scn_input))
        scnns = []
        for i in range(e_size):  # Size of ensemble
            scnn = small_cnn(
                scn_input, num_conv, keep_prob, name='scnn' + str(i))
            # scnn = tf.reshape(scnn, [m * n, -1, num_conv])
            # scnn = tf.reduce_min(scnn, 0)
            scnns.append(scnn)
            print('[ensemble_member] output: {}'.format(scnn))
            pass

        # scnn_e = tf.concat(scnns, 0)
        scnn_e = tf.stack(scnns, 0)
        # scnn_e=tf.reduce_mean(scnn_e,0)
        # print('[concated_ensemble] output: {} size: {}'.format(
        # scnn_e, scnns.__len__()))

        scnn_e = tf.reshape(scnn_e, [m * n, -1, num_conv * e_size])
        scnn_e = tf.reduce_mean(scnn_e, 0)
        output = scnn_e
        print('[scsnn_output]: {}'.format(output))

    return output, keep_prob


def main(_):
    # Read data from MNIST
    # mnist = input_data.read_data_sets('fashion', one_hot=True)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Placehoder of input and output
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='input')

        y_ = tf.placeholder(tf.float32, [None, 10], name='validation')

    # The main model
    e_size = 1
    y_conv, keep_prob = scscn(x, 1, 10, e_size=e_size)

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

    with tf.name_scope('config'):
        config = tf.ConfigProto(
            inter_op_parallelism_threads=256, intra_op_parallelism_threads=64)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.name_scope('logger'):
        # Graph
        run_description = 'l2_lrelu_aug_conv1d' + str(e_size)
        import time

        graph_location = '/tmp/saved_models/' + run_description  #+ str(time.time())
        print('Saving graph to: %s' % graph_location)

        writer = tf.summary.FileWriter(
            graph_location, graph=tf.get_default_graph())

        saver = tf.train.Saver()
        save_location = '/tmp/saved_models/' + run_description + '/saved'
        recover_location = '/tmp/saved_models/' + run_description + '/'

        # Loss
        tf.summary.scalar("t_loss", cross_entropy)
        tf.summary.scalar("t_acc", accuracy)
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
        # rt = 1e-2
        train_loss = 0
        for i in range(140001):
            bs = 64
            # Get the data of next batch
            batch = mnist.train.next_batch(bs)
            if i % 1000 == 0:
                rt = rt * 0.95
                if i == 600000:
                    rt = 3e-4
                    print('new rt: {}'.format(rt))
                if i == 1000000:
                    rt = 1e-4
                    print('new rt: {}'.format(rt))

                # Print the accuracy
                train_accuracy = 0
                validation_loss = 0

                for index in range(50):
                    vbs = 200
                    accuracy_batch = mnist.test.next_batch(vbs)
                    new_acc, v_loss = sess.run(
                        [accuracy, cross_entropy],
                        feed_dict={
                            x: accuracy_batch[0],
                            y_: accuracy_batch[1],
                            keep_prob: 0.5,
                        })
                    train_accuracy += new_acc
                    validation_loss += v_loss

                train_accuracy /= 50
                validation_loss /= 50
                overfit = (validation_loss / vbs - train_loss / bs) * 100
                print(
                    'epoch: %g|acc: %g|time: %g|v_loss: %g|train_loss: %g|overfit: %g|lr: %g'
                    % (i, train_accuracy, (time.clock() - t0), validation_loss,
                       train_loss, overfit, rt))
                t0 = time.clock()

                # Log other
                summary = tf.Summary()
                summary.value.add(tag='acc_v', simple_value=train_accuracy)
                summary.value.add(tag='v_loss', simple_value=validation_loss)
                summary.value.add(tag='lr', simple_value=rt)
                summary.value.add(tag='overfit', simple_value=overfit)
                writer.add_summary(summary, i)

                # Log loss
                summary = summary_op.eval(feed_dict={
                    x: accuracy_batch[0],
                    y_: accuracy_batch[1],
                    keep_prob: 1.0
                })
                writer.add_summary(summary, i)
                # Save parameters
                if (i % 5000 == 0):
                    real_location = saver.save(
                        sess, save_location, global_step=999999
                    )  # Making sure it's the lastest_checkpoint
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
