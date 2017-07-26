from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# Small CNN:
# convolution fliter of SCSCN
def small_cnn(x,num_conv):
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, x.get_shape().as_list()[3], num_conv])
    b_conv1 = bias_variable([num_conv])
    h_conv1 = tf.nn.relu(conv2d_(x, W_conv1) + b_conv1)
    
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, num_conv, num_conv])
    b_conv2 = bias_variable([num_conv])
    h_conv2 = tf.nn.relu(conv2d_(h_conv1, W_conv2) + b_conv2)

  # with tf.name_scope('max_pool'):
  #   h_pool = max_pool_2x2(h_conv)

  with tf.name_scope('avg_pool'):
    y_fc = avg_pool(h_conv2,h_conv2.get_shape().as_list()[1],
                h_conv2.get_shape().as_list()[2])

  return y_fc

def deepnn(x):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  with tf.name_scope('SCSCN'):
    h_scscn= scscn(x_image, 8, 4)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_scscn, W_conv2) + b_conv2)

  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([5 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def avg_pool(x, m, n):
  return tf.nn.avg_pool(x, ksize=[1, m, n, 1],
                        strides=[1, m, n, 1], padding='SAME')


def scscn(x, num, num_conv):
  with tf.name_scope('kernal_size'):
    # Kernal size:
    a=14
    b=14

  with tf.name_scope('strides'):
    # Strides:
    stride=2
  
  with tf.name_scope('pad'):
    # pad of input
    x = tf.pad(x, [[0, 0], [stride, stride], [stride, stride], [0, 0]])

  with tf.name_scope('input_size'):
    # Size of input:
    input_num=x.get_shape().as_list()[3]
    input_m=x.get_shape().as_list()[1]
    input_n=x.get_shape().as_list()[2]
    print('[input|SCSCN]num %d,m %d,n %d' % (input_num, input_m, input_n))

  with tf.name_scope('size'):
    # Size:
    m=int((input_m-a)/stride+1)
    n=int((input_n-b)/stride+1)

  with tf.name_scope('output'):
    # Output:
    # a TensorArray of tensor used to storage the output of small_cnn
    output=tf.TensorArray(tf.float32,
              int(num*m*n))

  with tf.name_scope('fliter'):
    for i in range(num):
      print('[init|SCSCN]i %d' % (i))
      for j in range(m):
        for k in range(n):
          # calculate the output of the convolution fliter
          output = output.write(int((j*n+k)*num+i),
                       small_cnn(tf.slice(x, [0,j*stride,k*stride,0],
                                   [-1,a,b,input_num]),num_conv))

  # return the concated and reshaped data of output
  return tf.reshape(output.concat(),
              [-1,m,n,num*num_conv])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Read data from MNIST
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  # Placehoder of input and output
  x = tf.placeholder(tf.float32, [None, 784])

  y_ = tf.placeholder(tf.float32, [None, 10])

  # The main model
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  # Start to run
  with tf.Session() as sess:
    # TensorBoard
    # summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      # Get the data of next batch
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        # Print the accuracy
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        # TensorBoard
        # summary_writer.add_summary(,i)
      # print('step %d' % (i))
      # Train
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
