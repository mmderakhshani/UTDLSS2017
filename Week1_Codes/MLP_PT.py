import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


# Import MNIST data
root = './data'
download = True
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True,
                       transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False,
                      transform=trans)

# Learning Parameters
training_epochs = 10
batch_size = 100
learning_rate = 0.1


# Network Parameters
n_input = 28*28
n_hidden_1 = 512
n_hidden_2 = 64
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
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_classes)
    def forward(self, x):
        x = x.view(-1, n_input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    def name(self):
        return 'mlpnet'


# Create The Model and Optimizer
model = MLPNet()
print(model)
ceriation = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training Epochs
for epoch in range(training_epochs):
    avg_loss = 0
    train_accuracy = 0
    # Training
    for x, target in train_loader:
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = ceriation(out, target)
        loss.backward()
        optimizer.step()
        avg_loss += loss.data[0]/len(train_loader)
        _, pred_label = torch.max(out.data, 1)
        train_accuracy += \
            (pred_label == target.data).sum()/len(train_loader)

    print("Epoch:", epoch+1, "Train Loss:", avg_loss,
          "Train Accuracy", train_accuracy)

    # Test
    test_accuracy = 0
    for x, target in test_loader:
        x, target = Variable(x, volatile=True),\
                    Variable(target, volatile=True)
        out= model(x)
        _, pred_label = torch.max(out.data, 1)
        test_accuracy += \
            (pred_label == target.data).sum() / len(test_loader)
    print("Test Accuracy", test_accuracy)