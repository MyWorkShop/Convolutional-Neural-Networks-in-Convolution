from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import numpy as np
import random
import math
import logging
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
random.seed(a=None)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Read data from MNIST
# mnist = input_data.read_data_sets('fashion', one_hot=True)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Training Setting
rt = 1e-3
aug = False
bs = 96

use_bn = True

# LSUV Setting
use_lsuv = False
tol = 0.2
t_max = 30
dry_run = True

e_size = 1  # Ensemble size
run_description = 'bn_swish_conv2_fc256_128_dp3_0.5_bs' + str(
    bs) + '_lr_' + str(rt) + 'normal' + str(e_size)
graph_location = '/tmp/saved_models/' + run_description  #+ str(time.time())
save_location = '/tmp/saved_models/' + run_description + '/saved'
recover_location = '/tmp/saved_models/' + run_description + '/'
if dry_run:
    run_description = 'Dry_' + run_description
    graph_location = '/tmp/saved_models/' + run_description + str(time.time())
    save_location = '/tmp/saved_models/' + run_description + '/saved'
    recover_location = '/tmp/saved_models/' + run_description + '/'
values_to_log = []

with tf.name_scope('config'):
    config = tf.ConfigProto(
        # inter_op_parallelism_threads=256, intra_op_parallelism_threads=64)
    )
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess_config = config

regularizer = tf.contrib.layers.l2_regularizer(0.2)
custom_loss=0