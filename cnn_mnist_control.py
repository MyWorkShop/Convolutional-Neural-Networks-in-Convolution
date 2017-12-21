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

# Classical CNN for contrast in overfit index


def CNN(x, reuse=False):
    print('CNN input: {}'.format(x))
    with tf.variable_scope('CNN', reuse=reuse):
        with tf.name_scope('reshape1'):
            x = tf.reshape(x, [-1, 28, 28, 1])

        with tf.name_scope('conv1'):
            x = tf.layers.conv2d(x, 20, [5, 5], padding='SAME')

        with tf.name_scope('pool1'):
            x = tf.layers.max_pooling2d(x, [2, 2], [2, 2], padding='SAME')

        with tf.name_scope('conv2'):
            x = tf.layers.conv2d(x, 50, [5, 5], padding='SAME')

        with tf.name_scope('pool2'):
            x = tf.layers.max_pooling2d(x, [2, 2], [2, 2],padding='VALID')
            #=>[-1,7,7,50]
        print('Convolution output: {}'.format(x))

        with tf.name_scope('reshape2'):
            x = tf.reshape(x, [-1, 7*7*50])

        with tf.name_scope('dense1'):
            x = tf.layers.dense(x, 256, tf.nn.elu)

        with tf.name_scope('dense2'):
            x = tf.layers.dense(x, 64, tf.nn.elu)

        with tf.name_scope('dense3'):
            x = tf.layers.dense(x, 10, tf.nn.elu)

        print('CNN output: {}'.format(x))
        return x
    pass


def main(_):
    # Read data from MNIST
    # mnist = input_data.read_data_sets('fashion', one_hot=True)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Placehoder of input and output
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='validation')

    with tf.name_scope('CNN'):
        y_conv = CNN(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y_conv)

        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        cross_entropy = tf.reduce_mean(
            cross_entropy)  #+ tf.reduce_mean(reg_losses)

    with tf.name_scope('adam_optimizer'):
        rate = tf.placeholder(tf.float32)
        train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)
    #'''
    with tf.name_scope('momentum_optimizer'):  #this works really bad...
        train_step_mmntm = tf.train.MomentumOptimizer(
            rate, momentum=0.9).minimize(cross_entropy)
    #'''

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        #accuracy = dummy = tf.constant(1)

    with tf.name_scope('config'):
        config = tf.ConfigProto(
            inter_op_parallelism_threads=256, intra_op_parallelism_threads=64)
        config.gpu_options.allow_growth = True

    with tf.name_scope('logger'):
        # Graph
        run_description = 'Normal_CNN'
        import time

        graph_location = '/tmp/saved_models/' + run_description  #+ str(time.time())
        print('Saving graph to: %s' % graph_location)
        writer = tf.summary.FileWriter(
            graph_location, graph=tf.get_default_graph())

        saver = tf.train.Saver()
        save_location = '/tmp/saved_models/' + run_description + '/saved'
        recover_location = '/tmp/saved_models/' + run_description + '/'

        # Loss
        tf.summary.scalar('loss', cross_entropy)
        summary_op = tf.summary.merge_all()

    #'''
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

        import numpy as np
        print('[total_trainable]: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        t0 = time.clock()
        rt = 0.001
        train_loss = 0
        #for i in range(150000):
        for i in range(150000):
            # Get the data of next batch
            bs = 16
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
                    vbs=200
                    accuracy_batch = mnist.test.next_batch(vbs)
                    new_acc, v_loss = sess.run(
                        [accuracy, cross_entropy],
                        feed_dict={
                            x: accuracy_batch[0],
                            y_: accuracy_batch[1],
                        })
                    train_accuracy += new_acc
                    validation_loss += v_loss
                    pass
                train_accuracy /= 50
                validation_loss /= 50

                print(
                    'epoch: %g|acc: %g|time: %g|v_loss: %g|train_loss: %g|overfit: %g|'
                    % (i, train_accuracy, (time.clock() - t0), validation_loss,
                       train_loss, (validation_loss/vbs - train_loss/bs)*100))
                t0 = time.clock()

                # Log loss
                summary = summary_op.eval(feed_dict={
                    x: accuracy_batch[0],
                    y_: accuracy_batch[1],
                })
                writer.add_summary(summary, i * bs)

                # Save parameters
                if (i % 5000 == 0):
                    real_location = saver.save(
                        sess, save_location, global_step=999999)
                    print('[saver] Model saved at {}'.format(real_location))
                    pass
                pass

            # Train
            _, train_loss = sess.run(
                [train_step, cross_entropy],
                feed_dict={
                    x: batch[0],
                    y_: batch[1],
                    rate: rt
                })
    #'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
