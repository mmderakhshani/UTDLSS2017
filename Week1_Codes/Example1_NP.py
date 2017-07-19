import numpy as np

w = np.array([[1, -1, 1], [0, 1, -1]])
b = np.array([[-2], [0]])

x = np.array([[2], [0], [2]])
y_hat = np.array([[1], [1]])

a = w @ x
y = a + b
e = y - y_hat
l = np.sum(e**2)

grad_l = 1.0
grad_e = grad_l * (2.0 * e)
grad_y = grad_e.copy()
grad_b = grad_y.copy()
grad_a = grad_y.copy()
grad_w = grad_a @ x.transpose()

print('y\n', y)
print('loss:\n', l)
print('grad_b:\n', grad_b)
print('grad_w:\n',grad_w)