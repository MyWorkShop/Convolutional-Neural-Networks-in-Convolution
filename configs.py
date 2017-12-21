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

import tensorflow as tf

FLAGS = None
random.seed(a=None)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

e_size = 1
run_description = 'bn_swish_fc1_32_fc256_128_dp3_0.5_bs48_lr_e3normal' + str(
    e_size)
graph_location = '/tmp/saved_models/' + run_description  #+ str(time.time())
save_location = '/tmp/saved_models/' + run_description + '/saved'
recover_location = '/tmp/saved_models/' + run_description + '/'
rt = 1e-3
aug = False
bs = 48
