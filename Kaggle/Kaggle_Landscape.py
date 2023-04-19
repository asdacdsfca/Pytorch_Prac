import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import glob
import torchvision
import timm

data_dir = './data/Landscape/Landscape Classification/' 
print(os.listdir(data_dir))
train_dir = data_dir+'Training Data/'
print(os.listdir(train_dir))
train_image_files = glob.glob(train_dir + '*/*.jpeg', recursive=True)
print(len(train_image_files))
print(train_image_files[0:10])

valid_dir = data_dir+'Validation Data/'
valid_image_files = glob.glob(valid_dir + '*/*.jpeg', recursive=True)
len(valid_image_files)

testing_dir = data_dir+'Testing Data/'
testing_image_files = glob.glob(testing_dir + '*/*.jpeg', recursive=True)
len(testing_image_files)

class_names = [f for f in os.listdir(train_dir)]
num_class = len(class_names)

idx_to_class = {i:j for i, j in enumerate(class_names)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
print(idx_to_class)
print(class_to_idx)

def show(train_image_files):
    plt.subplots(2, 2, figsize=(10,10))
    for i, k in enumerate(np.random.randint(len(train_image_files), size=4)):
        im = Image.open(train_image_files[k])
        arr = np.array(im)
        plt.subplot(2, 2, i+1)
        plt.axis('off')
        img_name = train_image_files[k].split('/')[-1]
        img_label = train_image_files[k].split('/')[-2]
        plt.title(f"{img_name}, {img_label}, {im.size}")
        plt.imshow(arr, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

# Setting up the basic functions/methods
class LandscapeDataset(Dataset):
    # initialization/constructor
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    # create a method that returns the length of the image
    def __len__(self):
        return len(self.image_paths)
    
    def __getattrs__(self, idx):
        file_paths = self.image_paths[index]
        image = cv2.imread(file_paths)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# We can check the size of the image by the following lines:
# and the result should be 224
print(timm.data.IMAGENET_DEFAULT_MEAN)
print(timm.data.IMAGENET_DEFAULT_STD)

# pre-trained model for trainning data
train_transforms = transforms.Compose([
    # python interpretable
    transforms.ToPILImage(),
    # because the mean size our image is 224, we'd like
    # our model to be trained on that size
    transforms.RandomResizedCrop(224,224),
    # apply some sort of transformation to help trainning
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ore-trained model for validation data
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = LandscapeDataset(train_image_files, train_transforms)
val_dataset = LandscapeDataset(valid_image_files, val_transforms)
test_dataset = LandscapeDataset(testing_image_files, val_transforms)

def loaddata(dataset, batch):
    if dataset == train_dataset:
        DataLoader(dataset, batch_size = batch, shuffle = True)
    else:
        DataLoader(dataset, batch_size = batch)

train_load = loaddata(train_dataset, 32)
val_load = loaddata(val_dataset, 32)
test_load = loaddata(test_dataset, 32)

def clustering():
    
