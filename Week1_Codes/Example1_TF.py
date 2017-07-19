import numpy as np
import tensorflow as tf

w = tf.Variable(np.array([[1, -1, 1], [0, 1, -1]]), dtype = tf.float32)
b = tf.Variable(np.array([[-2], [0]]), dtype= tf.float32)
x = tf.placeholder(tf.float32)
y_hat = tf.placeholder(tf.float32)
values = {
        x: np.array([[2], [0], [2]]),
        y_hat: np.array([[1], [1]])
    }

with tf.device('/gpu:0'):
    a = tf.matmul(w,x)
    y = a + b
    l = tf.reduce_sum((y-y_hat)**2 )
    grad_b, grad_w = tf.gradients(l, [b, w])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    grad_w_val, grad_b_val, y_val, l_val = sess.run([grad_w, grad_b, y, l], feed_dict=values)

print('y\n', y_val)
print('loss:\n', l_val)
print('grad_b:\n', grad_b_val)
print('grad_w:\n',grad_w_val)