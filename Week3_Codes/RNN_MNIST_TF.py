import tensorflow as tf
import numpy as np
import time

np.random.seed(1)
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot = True)

# Learning Parameters
training_epochs = 20
batch_size = 100
learning_rate = 0.01

# Network Parameters
n_input = 28*28
seq_length = 28
feature_size = 28
state_size = 50
n_classes = 10

# TF Graph Input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
with tf.device('cpu:0'):
    tf.set_random_seed(1)
    # Reshape Input
    input_layer = tf.reshape(x, [-1, seq_length, feature_size])

    rnn_cell = tf.contrib.rnn.BasicRNNCell(
        num_units=state_size,
        activation=tf.nn.relu)
    # rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob =1-dropout_prob)
    rnn_outputs, _ = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=input_layer, dtype=tf.float32)
    print(rnn_outputs)
    rnn_last_output = rnn_outputs[:,-1]
    print(rnn_last_output)
    preds = tf.layers.dense(
        inputs=rnn_last_output,
        units=n_classes)

    loss = tf.losses.softmax_cross_entropy(logits=preds,
                                           onehot_labels=y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
        .minimize(loss)

    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# Run The Graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    start_process = time.time()
    for epoch in range(training_epochs):
        start_epoch = time.time()
        avg_loss = 0
        train_accuracy = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, batch_loss, batch_accuracy = \
                sess.run([optimizer, loss, accuracy],
                         feed_dict={x: batch_x,y: batch_y})
            avg_loss += batch_loss/total_batch
            train_accuracy += batch_accuracy/total_batch
        end_epoch = time.time()
        print("Epoch:", epoch+1, "Train Loss:",
              avg_loss, "Train Accuracy", train_accuracy,
              "in:", int(end_epoch - start_epoch), "sec")
        # Test
        test_accuracy = \
            sess.run([accuracy],
                     feed_dict={x: mnist.test.images,
                                y: mnist.test.labels})
        print("Test Accuracy", test_accuracy)
    end_process = time.time()
    print("Train (& test) completed in:",
    int(end_process - start_process), "sec")