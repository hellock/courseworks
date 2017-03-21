from __future__ import print_function
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cvbase.io import pickle_dump
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
use_cuda = True
seed = int(time.time())
epochs = 10
log_interval = 1000
batch_size = 4
test_batch_size = 4
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4

torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, **kwargs)

testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=test_batch_size, shuffle=False, **kwargs)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.msra_init()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 400)
        x = self.fc1(x)
        # x = self.bn3(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.bn4(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return F.log_softmax(x)
        # x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = x.view(-1, 400)
        # x = F.relu(self.bn3(self.fc1(x)))
        # x = F.relu(self.bn4(self.fc2(x)))
        # x = F.relu(self.fc3(x))
        # return F.log_softmax(x)

    def msra_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2.0 / n))


model = Net()
if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(),
                      lr=lr,
                      momentum=momentum,
                      weight_decay=weight_decay)

losses = []


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # is it true to use such a loss over cross-entropy loss?
        loss = F.nll_loss(output, target)
        losses.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(
                    train_loader), loss.data[0]))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(
        test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
          format(test_loss, correct,
                 len(test_loader.dataset), 100. * correct / len(
                     test_loader.dataset)))


for epoch in range(1, epochs + 1):
    train(epoch)
params = [param.data.cpu().numpy() for param in model.parameters()]
pickle_dump(params, 'params.pkl')
losses = np.concatenate(losses)
pickle_dump(losses, 'loss.pkl')
test(epoch)
