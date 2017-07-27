import tensorflow as tf
import numpy as np
import time

np.random.seed(1)
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot = True)

# Learning Parameters
training_epochs = 10
batch_size = 100
learning_rate = 0.01

# Network Parameters
n_input = 28*28
n_classes = 10

# TF Graph Input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
with tf.device('cpu:0'):
    tf.set_random_seed(1)
    # Reshape Input
    input_layer = tf.reshape(x, [-1, 28, 28, 1])
    # Define Convolution(& Pooling) Layer1
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters = 16,
                             kernel_size=[5, 5],
                             strides=[1,1],
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2,2],
                                    strides=2)

    # Define Convolution(& Pooling) Layer2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters = 64,
                             kernel_size=[5, 5],
                             strides=[1, 1],
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                pool_size=[2, 2],
                                strides=2)

    # Multiply The Shape of Previous Layer
    dim = np.prod(pool2.get_shape().as_list()[1:])

    # Define Fully Connected Layers
    relu2_flat = tf.reshape(pool2, [-1, dim])
    fc1 = tf.layers.dense(inputs=relu2_flat,
                          units=256,
                          activation=tf.nn.relu)
    fc2 = tf.layers.dense(inputs=fc1,
                          units=n_classes)

    # Define Loss and Optimizer
    loss = tf.losses.softmax_cross_entropy(logits=fc2,
                                           onehot_labels=y)
    optimizer = tf.train.\
        GradientDescentOptimizer(learning_rate=learning_rate)\
        .minimize(loss)

    # Define Accuracy
    correct_pred = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
