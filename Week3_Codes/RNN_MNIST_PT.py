import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
import pdb

torch.manual_seed(1234)

# Import MNIST data
root = './data'
download = False
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True,
                       transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False,
                      transform=trans)

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

# Data Loader
kwargs = {'num_workers': 1}
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False, **kwargs)


# Define The Model
class VanillaRNN(nn.Module):
    def __init__(self):
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size=feature_size,
                          hidden_size=state_size,
                          nonlinearity='relu')
        self.fc1 = nn.Linear(state_size, n_classes)
    def forward(self, x):
        _, last_hidden = self.rnn(x)
        out = self.fc1(last_hidden.squeeze(0))
        return out
    def name(self):
        return 'VanillaRNN'

# Create The Model and Optimizer
model = VanillaRNN()
print(model)
ceriation = nn.CrossEntropyLoss()
optimizer = optim.SGD([{'params':model.rnn.parameters()},
					  {'params':model.fc1.parameters()}],
					lr=learning_rate)

# Training Epochs
start_process = time.time()
for epoch in range(training_epochs):
    start_epoch = time.time()
    avg_loss = 0
    train_accuracy = 0
    # Training
    for x, target in train_loader:
        optimizer.zero_grad()
        x, target = Variable(x).squeeze(1).permute(1,0,2), Variable(target)
        out = model(x)
        loss = ceriation(out, target)
        loss.backward()
        optimizer.step()
        avg_loss += loss.data[0]/len(train_loader)
        _, pred_label = torch.max(out.data, 1)
        train_accuracy += \
             (pred_label == target.data).sum()/len(train_loader)
    end_epoch = time.time()
    print("Epoch:", epoch+1, "Train Loss:", avg_loss,
          "Train Accuracy", train_accuracy,
          "in:", int(end_epoch-start_epoch), "sec")
    # Test
    test_accuracy = 0
    for x, target in test_loader:
        x, target = Variable(x).squeeze(1).permute(1,0,2), Variable(target)
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        test_accuracy += \
            (pred_label == target.data).sum() / len(test_loader)
    print("Test Accuracy", test_accuracy)
end_process = time.time()
print ("Train (& test) completed in:",
       int(end_process-start_process), "sec")
