#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from model import *
from configs import *

# TODO:

# Small CNN:
# convolution fliter of SCSCN


def main(_):
    global mnist
    global scnscn

    #"""
    # Start to run
    with tf.Session(config=sess_config) as sess:

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
        train_loss = 0
        # Image augmentation
        dgen = tf.contrib.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=None,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest')
        for i in range(60001):

            # Get the data of next batch
            batch = mnist.train.next_batch(bs)
            # Optional noise(1/2 chance if enabled)
            if aug and bool(random.randint(0, 1)):
                batch = dgen.flow(batch, batch_size=bs)
                pass

            global rt
            if i % 900 == 0:
                # rt *= 0.993
                # '''
                if i == 30000:
                    rt = 3e-4
                    print('new rt: {}'.format(rt))
                if i == 42000:
                    rt = 9e-5
                    print('new rt: {}'.format(rt))
                if i == 54000:
                    rt = 3e-5
                    print('new rt: {}'.format(rt))
                # '''

                # Print the accuracy
                validation_accuracy = 0
                validation_loss = 0

                vbs = 10
                itr = 1000
                for index in range(itr):
                    accuracy_batch = mnist.test.next_batch(vbs)
                    new_acc, v_loss = sess.run(
                        [accuracy, cross_entropy],
                        feed_dict={
                            x: accuracy_batch[0],
                            y_: accuracy_batch[1],
                            keep_prob: 1,
                            phase_train: False
                        })
                    validation_accuracy += new_acc
                    validation_loss += v_loss

                validation_accuracy /= itr
                validation_loss /= itr
                overfit = (validation_loss / vbs - train_loss / bs) * 100
                print((
                    '\x1b[6;30;42m' +
                    'epoch: %g|acc: %g|time: %g|v_loss: %g|train_loss: %g|overfit: %g|lr: %g'
                    + '\x1b[0m') %
                      (i, validation_accuracy, (time.clock() - t0),
                       validation_loss, train_loss, overfit, rt))
                t0 = time.clock()

                # Log other
                summary = tf.Summary()
                summary.value.add(
                    tag='acc_v', simple_value=validation_accuracy)
                summary.value.add(tag='v_loss', simple_value=validation_loss)
                summary.value.add(tag='lr', simple_value=rt)
                summary.value.add(tag='overfit', simple_value=overfit)
                writer.add_summary(summary, i)

                # Log loss
                summary = summary_op.eval(
                    feed_dict={
                        x: accuracy_batch[0],
                        y_: accuracy_batch[1],
                        keep_prob: 1.0,
                        phase_train: False
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
                    rate: rt,
                    phase_train: False
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
