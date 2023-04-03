import torch
from torch import nn
from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
batch_size = 64

# Load data
train_dataset = datasets.MNIST(root='./data/', 
                               download=True, 
                               train=True, 
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data/', 
                               download=True, 
                               train=False, 
                               transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               shuffle=True, 
                                               batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                               shuffle=False, 
                                               batch_size=batch_size)

# Basic analysis about the data

# see the data shape
print(train_dataset.data.shape, train_dataset.targets.shape)
print(test_dataset.data.shape, test_dataset.targets.shape)

# Look at the min and the max
print(train_dataset.data.min(), train_dataset.data.max())

# transfer to float to see the mean
print(train_dataset.data.float().mean(), train_dataset.data.float().median())

# classes in dataset
train_dataset.classes

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

        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=16, 
                                            kernel_size=5, padding=2, stride=1), 
                                  nn.BatchNorm2d(16), 
                                  nn.ReLU(), 
                                  nn.MaxPool2d(kernel_size=2, stride=2)) #change it from 28*28 to 44*44
        

        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, 
                                            kernel_size=5, padding=2, stride=1), 
                                  nn.BatchNorm2d(32), 
                                  nn.ReLU(), 
                                  nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(32*(input_shape//4)*(input_shape//4), num_classes)

    
    def forward(self, x):

        out = self.cnn1(x)

        out = self.cnn2(out)

        # flattens
        # so feature maps extracted by the convolutional layers into a format that can be fed into the fc.
        out = out.reshape(out.size(0), -1)
        # Applies the fc to tensor
        out = self.fc(out)
        return out

model = CNN(input_shape=input_shape, num_classes=num_classes, in_channels=1).to(device)

# cpu/gpu
# note that cpu is used when writing this, though it doesn't matter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# As expected
summary(model, input_size=(1, 28, 28), batch_size=batch_size)

# Hyperparameters
epochs = 3
learning_rate = 0.01

# Define our loss
criterion = nn.CrossEntropyLoss()
optimzer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(train_dataloader)
print(f'The total batch is {total_batch}')

# train the model
for epoch in range(epochs):
    # loop batches
    for batch_idx, (images, labels) in enumerate(train_dataloader):

        images = images.to(device)
        labels = labels.to(device)
        
        # forward: for the output and loss
        out = model(images)
        loss = criterion(out, labels)
        
        # backward: update the step size by gradidents
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
print(f'{correct/total}')