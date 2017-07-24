from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def small_cnn(x,W_fc,b_fc):
  with tf.name_scope('conv1'):
    W_conv = weight_variable([5, 5, x.get_shape().as_list()[3], 8])
    b_conv = bias_variable([8])
    h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)

  with tf.name_scope('pool1'):
    h_pool = max_pool_2x2(h_conv)

  with tf.name_scope('fc1'):
    h_pool_flat = tf.reshape(h_pool, [-1, 3 * 3 * 8])
    y_conv = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)

  return y_conv

def deepnn(x):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # with tf.name_scope('conv1'):
  #   W_conv1 = weight_variable([5, 5, 1, 32])
  #   b_conv1 = bias_variable([32])
  #   h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # with tf.name_scope('pool1'):
  #   h_pool1 = max_pool_2x2(h_conv1)

  with tf.name_scope('SCSCN'):
    h_scscn= scscn(x_image,32)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.relu(conv2d(h_scscn, W_conv2) + b_conv2)

  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
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


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def scscn(x,num):
  # Kernal size:
  a=10
  b=10
  # Size of input:
  input_num=x.get_shape().as_list()[3]
  input_m=x.get_shape().as_list()[1]
  input_n=x.get_shape().as_list()[2]
  print('[input|SCSCN]num %d,m %d,n %d' % (input_num, input_m, input_n))
  # Strides:
  stride=2;
  # Output:
  # a TensorArray of tensor used to storage the output of small_cnn
  output=tf.TensorArray(tf.float32,
              int(num*((input_m-a)/stride+1)*((input_n-b)/stride+1)))
  # weight and bias of the fc layer of the small_cnn
  Weight_fc = tf.TensorArray(tf.float32,num)
  Bias_fc = tf.TensorArray(tf.float32,num)
  for i in range(num):
    Weight_fc.write(i,weight_variable([3 * 3 * 8, 1])).mark_used()
    Bias_fc.write(i,bias_variable([1])).mark_used()
    for j in range(int((input_m-a)/stride+1)):
      for k in range(int((input_n-b)/stride+1)):
        output.write(int((j*((input_n-b)/stride+1)+k)*num+i),
                     small_cnn(tf.slice(x,[0,j*stride,k*stride,0],[1,a,b,input_num]),
                                 Weight_fc.read(i),Bias_fc.read(i))).mark_used()
  # return the concated and reshaped data of output
  return tf.reshape(output.concat(),
              [1,int(((input_m-a)/stride+1)),int(((input_n-b)/stride+1)),num])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784])

  y_ = tf.placeholder(tf.float32, [None, 10])

  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
