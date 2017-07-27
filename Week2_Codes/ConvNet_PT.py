import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
torch.manual_seed(1)
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
training_epochs = 10
batch_size = 100
learning_rate = 0.1


# Network Parameters


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
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
        	                   out_channels=16,
            				   kernel_size=(5,5),
            				   stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=16,
            				   out_channels=64,
            				   kernel_size=(5,5),
            				   stride=(1,1))
        self.fc1 = nn.Linear(in_features=4*4*64,
            			     out_features=256)
        self.fc2 = nn.Linear(in_features=256,
            			     out_features=10)
    def forward(self,x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def name(self):
        return 'lenet'

# Create The Model and Optimizer
model = ConvNet()
print(model)
ceriation = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training Epochs
start_process = time.time()
for epoch in range(training_epochs):
    start_epoch = time.time()
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
    end_epoch = time.time()
    
    print("Epoch:", epoch+1, "Train Loss:", avg_loss,
          "Train Accuracy", train_accuracy,
          "in:", int(end_epoch-start_epoch), "sec")
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
end_process = time.time()
print ("Train (& test) completed in:",
       int(end_process-start_process), "sec")
