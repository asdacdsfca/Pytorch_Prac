import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

#MINST
train_dataset = torchvision.datasets.MNIST(root='../data', 
                                           train=True, 
                                           download=True, 
                                           transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='../data', 
                                          train=False, 
                                          download=True, 
                                          transform=transforms.ToTensor())

# data size
len(train_dataset)

# print(image)
plt.imshow(train_dataset[0][0].reshape((28,28)), cmap = 'gray')

# see the data shape
print(train_dataset.data.shape, train_dataset.targets.shape)
print(test_dataset.data.shape, test_dataset.targets.shape)

# Look at the min and the max
print(train_dataset.data.min(), train_dataset.data.max())

# transfer to float to see the mean
print(train_dataset.data.float().mean(), train_dataset.data.float().median())

# classes in dataset
train_dataset.classes

# Save memory, not loading the data in one instance
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=64, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=128, 
                                          shuffle=True)

images, labels = next(iter(train_loader))

# batch, channel, height, weight
print(images.shape, labels.shape)

# number of batches, which means the batch size of the last image is not 64
len(train_loader)

# show data
fig = plt.figure(figsize=(5, 5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    plt.imshow(images[i][0], cmap='gray', interpolation='none')
    plt.title("gt {}".format(labels[i]))
    plt.xticks([])
    plt.yticks([])

# CIFAR RGB
train_dataset = torchvision.datasets.CIFAR10(root='../data', 
                                           train=True, 
                                           download=True, 
                                           transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='../data', 
                                          train=False, 
                                          download=True, 
                                          transform=transforms.ToTensor())

print(test_dataset.data.shape)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=64, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=128, 
                                          shuffle=True)

images, labels = next(iter(train_loader))

fig = plt.figure(figsize=(5, 5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    plt.imshow(np.transpose(images[i], (1, 2, 0)), interpolation='none')
    plt.title("gt {}".format(labels[i]))
    plt.xticks([])
    plt.yticks([])