from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import time

# Deopout rate
RATE_DROPOUT = 0.4

#Fully convolution networks
def fcn(x, phase_train):
    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    # Convolutional layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    # Convolutional layer #3
    conv3 = tf.layers.conv2d_transpose(
        inputs=conv2,
        filters=32,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)
    return tf.layers.dropout(inputs=conv3, rate=RATE_DROPOUT, training=phase_train)

def model(x):
    #If training
    phase_train = tf.placeholder(tf.bool)
    x = tf.reshape(x, [-1, 28, 28, 1])

    #CNNIC layer #1
    with tf.variable_scope("cnnic_1"):
        layer1_wrap = wrap(x, 5, 5, 3, [-1, 16, 16, 1])
        layer1_unwrap = fcn(layer1_wrap, phase_train)
        layer1_output = unwrap(layer1_unwrap, 5, 5, 3, [-1, 16, 16, 32])

    #CNNIC layer #2
    # with tf.variable_scope("cnnic_2"):
    #     layer2_wrap = wrap(layer1_output, 5, 5, 3, [-1, 16, 16, 32])
    #     layer2_unwrap = fcn(layer2_wrap, phase_train)
    #     layer2_output = unwrap(layer2_unwrap, 5, 5, 3, [-1, 16, 16, 32])

    #Pooling layer #1
    pool1 = tf.layers.average_pooling2d(inputs=layer1_output, pool_size=[2, 2], strides=2)

    #CNNIC layer #3
    with tf.variable_scope("cnnic_3"):
        layer3_wrap = wrap(pool1, 4, 4, 2, [-1, 8, 8, 32])
        layer3_unwrap = fcn(layer3_wrap, phase_train)
        layer3_output = unwrap(layer3_unwrap, 4, 4, 2, [-1, 8, 8, 32])

    #Pooling layer
    pool = tf.layers.average_pooling2d(inputs=layer3_output, pool_size=[2, 2], strides=2)

    # Dense layer
    pool_flat = tf.reshape(pool, [-1, 7 * 7 * 32])
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=RATE_DROPOUT, training=phase_train)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits, phase_train

def wrap(x, m, n, stride, shape):
    slicing = tf.TensorArray('float32', m * n)
    for j in range(m):
        for k in range(n):
            slicing = slicing.write(
                j * n + k, tf.slice(x, [0, j * stride, k * stride, 0],
                            shape))
    sliced = tf.reshape(slicing.concat(), shape)
    slicing.close().mark_used()
    return sliced

def unwrap(x, m, n, stride, shape):
    adding = tf.TensorArray('float32', m * n)
    to_split = tf.reshape(x, tf.concat([[m * n], shape], 0))
    split = tf.TensorArray('float32', m * n).unstack(to_split)
    for j in range(m):
        for k in range(n):
            adding = adding.write(j * n + k, tf.pad(split.read(j * n + k), 
                    [[0, 0], [j * stride, (m - 1 - j) * stride],
                            [k * stride, (n - 1 - k) * stride], [0, 0]]))
    split.close().mark_used()
    added = tf.reduce_sum(adding.stack(), 0)
    adding.close().mark_used()
    return added


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    input_data = tf.placeholder(tf.float32, [None, 784])
    output_data = tf.placeholder(tf.int64, [None])
    y_model, phase_train= model(input_data)

    #Loss
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=output_data, logits=y_model)
    cross_entropy = tf.reduce_mean(cross_entropy)

    #Optimizer
    rate = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)

    #Accuracy
    correct_prediction = tf.equal(tf.argmax(y_model, 1), output_data)
    correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t0 = time.clock()
        rt = 1e-3
        for i in range(60001):
            # Get the data of next batch
            batch = mnist.train.next_batch(100)
            if (i % 600 == 0) and (i != 0):
                if i == 30000:
                    rt = 3e-4
                if i == 42000:
                    rt = 1e-4
                if i == 48000:
                    rt = 3e-5
                if i == 54000:
                    rt = 1e-5
                # Print the accuracy
                test_accuracy = 0
                test_accuracy_once = 0
                for index in range(200):
                    accuracy_batch = mnist.test.next_batch(50)
                    test_accuracy_once = sess.run(accuracy, feed_dict={
                        input_data: accuracy_batch[0],
                        output_data: accuracy_batch[1],
                        phase_train: False})
                    test_accuracy += test_accuracy_once
                    test_accuracy_once = 0
                print('%g, %g, %g' %
                      (i / 600, test_accuracy / 200, (time.clock() - t0)))
                t0 = time.clock()
            # Train
            _ = sess.run(
                train_step,
                feed_dict={input_data: batch[0],
                           output_data: batch[1],
                           phase_train: True,
                           rate: rt})

if __name__ == "__main__":
    tf.app.run()