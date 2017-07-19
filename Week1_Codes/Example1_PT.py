import torch
from torch.autograd import Variable

w = Variable(torch.Tensor([[1, -1, 1], [0, 1, -1]]).cuda(),requires_grad = True)
b = Variable(torch.Tensor([[-2], [0]]).cuda(), requires_grad = True)
x = Variable(torch.Tensor([[2], [0], [2]]).cuda(), requires_grad = False)
y_hat = Variable(torch.Tensor([[1], [1]]).cuda(), requires_grad = False)


a = w @ x
y = a + b
l = torch.sum((y-y_hat)**2)

l.backward()

print('y\n', y)
print('loss:\n', l)
print('grad_b:\n', b.grad)
print('grad_w:\n',w.grad)