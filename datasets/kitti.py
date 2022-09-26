"""KITTI
http://www.cvlibs.net/datasets/kitti/
including 249966 images with labels
The folder is orgnized as:
KITTI
    /training
        /image_2
            /000000.png
            ...
            /007480.png
KITTI_seg
    /training
        /image_2
            /000000_10.png
            ...
            /000199_10.png
        /semantic
            /000000_10.png
            ...
            /000199_10.png
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

class KITTIdataset(Dataset):
    """
    Args:
        Dataset (_type_): nn.dataset
    Return
        id,image,label = tensor (BCHW)
        image in [0, 255] float32

    """
    def __init__(self,root="./datas",dataset="KITTI",split = 'training',transforms = None):
        self.root = root
        self.dataset = dataset
        self.split = split
        self.filenames = sorted(os.listdir(os.path.join(root,dataset,split,"image_2")))
        self.transforms = transforms
    def __len__(self):
        return len(self.filenames)    
    def __getitem__(self, index):
        if self.dataset.endswith('seg'):
            

            image = Image.open(os.path.join(self.root,self.dataset,self.split,"image_2",self.filenames[index])).convert('RGB')
            # image = np.asarray(image.convert('RGB'), dtype=np.uint8)

            label = Image.open(os.path.join(os.path.join(self.root,self.dataset,self.split,"semantic"),self.filenames[index]))
            label = np.asarray(label, dtype=np.int64)
            if self.transforms is None:
                self.transforms = T.Compose([T.ToTensor(),
                                    T.Resize(256,512)])
                
                # normalize = T.Normalize( mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
                image = self.transforms(image)
                label = self.transforms(label)
            else:
                image = self.transforms(image)
            return index+1,image,label
        else:
            image = Image.open(os.path.join(self.root,self.dataset,self.split,"image_2",self.filenames[index]))
            image = np.asarray(image.convert('RGB'), dtype=np.uint8)

            if self.transforms is None:
                self.transforms = T.Compose([T.ToTensor(),
                                    T.Resize((128,256))])
                
                image = self.transforms(image)
            else:
                image = self.transforms(image)
            

            return index+1,image,image