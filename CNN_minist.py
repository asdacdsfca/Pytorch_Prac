import torch
from torch import nn
from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
batch_size = 64

# Load data
train_dataset = datasets.MNIST(root='../data/', 
                               download=True, 
                               train=True, 
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='../data/', 
                               download=True, 
                               train=False, 
                               transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               shuffle=True, 
                                               batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                               shuffle=False, 
                                               batch_size=batch_size)

images, labels = next(iter(train_dataloader))

# batch_size, channels, height, weight
images.shape

# data
fig = plt.figure(figsize=(5, 5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    plt.imshow(images[i][0], cmap='gray', interpolation='none')
    plt.title("gt {}".format(labels[i]))
    plt.xticks([])
    plt.yticks([])

# Hyperparameters
input_shape = 28
num_classes = 10

# IMPORTANT: check out the pdf to see the derivation of parameters
# Goal: increment Channel, decrease shape
# Same convolution used
class CNN(nn.Module):
    # Three layers: 2 CNN, 1 fc
    # in_channels = dataset
    def __init__(self, input_shape, in_channels, num_classes):
        super(CNN, self).__init__()
        # conv2d: (b, 1, 28, 28) => (b, 16, 28, 28)
        # maxpool2d: (b, 16, 28, 28) => (b, 16, 14, 14)
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=16, 
                                            kernel_size=5, padding=2, stride=1), 
                                  nn.BatchNorm2d(16), 
                                  nn.ReLU(), 
                                  nn.MaxPool2d(kernel_size=2, stride=2)) #change it from 28*28 to 44*44
        
        # conv2d: (b, 16, 14, 14) => (b, 32, 14, 14)
        # maxpool2d: (b, 32, 14, 14) => (b, 32, 7, 7)
        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, 
                                            kernel_size=5, padding=2, stride=1), 
                                  nn.BatchNorm2d(32), 
                                  nn.ReLU(), 
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        # (b, 32, 7, 7) => (b, 32*7*7)
        # (b, 32*7*7) => (b, 10)
        self.fc = nn.Linear(32*(input_shape//4)*(input_shape//4), num_classes)

    
    def forward(self, x):
        # (b, 1, 28, 28) => (b, 16, 14, 14)
        out = self.cnn1(x)
        # (b, 16, 14, 14) => (b, 32, 7, 7)
        out = self.cnn2(out)
        # (b, 32, 7, 7) => (b, 32*7*7)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN(input_shape=input_shape, num_classes=num_classes, in_channels=1).to(device)

# cpu/gpu
# note that cpu is used when writing this, though it doesn't matter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# As expected
summary(model, input_size=(1, 28, 28), batch_size=batch_size)

# Hyperparameters
num_epochs = 3
learning_rate = 1e-3

# Define our loss
criterion = nn.CrossEntropyLoss()
optimzer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(train_dataloader)

# train the model
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # forward
        out = model(images)
        loss = criterion(out, labels)
        
        # backward
        optimzer.zero_grad()
        loss.backward()
        optimzer.step() 

# Evaluation
total = 0
correct = 0
for images, labels in test_dataloader:
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)
    preds = torch.argmax(out, dim=1)
    
    total += images.size(0)
    correct += (preds == labels).sum().item()
print(f'{correct}/{total}={correct/total}')