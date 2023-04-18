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

class LandscapeDataset(Dataset):
    # initialization/constructor
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    # create a method that returns the length of the image
    def __len__(self):
        return len(self.image_paths)
    
    def __getattrs__(self, idx):
        
