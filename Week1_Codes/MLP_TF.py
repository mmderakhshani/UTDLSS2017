import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot = True)


# Learning Parameters
training_epochs = 10
batch_size = 100
learning_rate = 0.1

# Network Parameters
n_input = 28*28
n_hidden_1 = 512
n_hidden_2 = 64
n_classes = 10

# TF Graph Input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# Construct Model
with tf.device('/gpu:0'):
    # Define Layers
    hidden_1 = tf.layers.dense(inputs=x, units=n_hidden_1,
                               activation=tf.nn.relu)
    hidden_2 = tf.layers.dense(inputs=hidden_1, units=n_hidden_2,
                               activation=tf.nn.relu)
    pred = tf.layers.dense(inputs=hidden_2, units=n_classes,
                           activation=tf.nn.relu)

    # Define Loss and Optimizer
    loss = tf.losses.softmax_cross_entropy(logits=pred,
                                           onehot_labels=y)
    optimizer = tf.train.\
        GradientDescentOptimizer(learning_rate=learning_rate)\
        .minimize(loss)

    # Define Accuracy
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Run The Graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    for epoch in range(training_epochs):
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
        print("Epoch:", epoch, "Train Loss:",
              avg_loss, "Train Accuracy", train_accuracy)
        # Test
        test_accuracy = \
            sess.run([accuracy],
                     feed_dict={x: mnist.test.images,
                                y: mnist.test.labels})
        print("Test Accuracy", test_accuracy)



