"""GTA5 datasets
https://download.visinf.tu-darmstadt.de/data/from_games/
including 249966 images with labels
The folder is orgnized as:
root
    /images
        /00001.png
        ...
        /24966.png
    /labels
        /00001.png
        ...
        /24966.png
images:
    resized to 256x128
labels:
    1 channel images
    class labels are compatible with the CamVid and CityScapes
    
"""
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np

class gta5dataset(Dataset):
    def __init__(self,root="./datas",dataset="GTA5",transforms = None):
        self.root = root
        self.dataset = dataset
        self.filenames = sorted(os.listdir(os.path.join(root,dataset,"images256")))
        self.transforms = transforms
    def __len__(self):
        return len(self.filenames)    
    def __getitem__(self, index):

            image = Image.open(os.path.join(os.path.join(self.root,self.dataset,"images256"),self.filenames[index]))
            image = np.asarray(image.convert('RGB'),dtype=np.float32)
            label = Image.open(os.path.join(os.path.join(self.root,self.dataset,"labels256"),self.filenames[index]))
            label = np.asarray(label, dtype=np.int64)
            if self. transforms is None:
                toTensor = T.ToTensor()
                normalize = T.Normalize((0,0,0),(1,1,1))
                image = normalize(toTensor(image))
            return index+1,image,label


