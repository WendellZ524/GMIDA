import torch
import torchvision
from models import DeepLabV3
from datasets import gta5dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.utils import make_grid
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import os
from tqdm import tqdm
from utils import colorize_mask
from PIL import Image
import cv2
def train():
    vis_transform = T.Compose([T.ToTensor()])
    dataset = gta5dataset("./datas/")
    # Data
    dataloader = DataLoader(dataset,
                                    shuffle=True,
                                    batch_size=2,
                                    num_workers=0,
                                    pin_memory=False)

    device = torch.device("cuda:0")
    model = DeepLabV3().to(device)      

    epochs = 20
    # train loop
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.01, momentum = 0.9 , weight_decay = 1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1 )

    # loss
    def compute_loss(output,target):
        ce = nn.CrossEntropyLoss(ignore_index=255)
        ce_loss = ce(output,target)
        return ce_loss

    # output dir
    run_time = time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())
    output_dir = "./train-runs"
    output_dir = os.path.join(output_dir,run_time)
    if os.path.isdir(output_dir):
        os.makedir(output_dir)
    # Tensorboard
    writer = SummaryWriter(os.path.join(output_dir,"log"))



    # Train loop
    interval = 500
    for epoch in range(epochs):
        # one eopch  
        model.train()
        print(f"epoch {epoch} starts at {time.strftime(r'%Y-%m-%d_%H-%M-%S',time.localtime())}")
        for i,(index,image,label) in tqdm(enumerate(dataloader)):
            output = model(image.to(device))['out']

            loss = compute_loss(output,label.to(device))
            # optimization step:
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)
            
            step = epoch*len(dataloader)+i 
            if step % interval ==0:
                writer.add_scalar("total_loss",loss,step)
            break
        # validate: visualize one image 
        model.eval()
        restore_transform = T.Compose([
                            T.Normalize(mean=[-123.675/58.395, -116.28/57.12, -103.53/57.375], std=[1./58.395, 1./57.12, 1./57.375]),
                            T.ToPILImage()])

        im = image[0].data.cpu()
        im = np.asarray(restore_transform(im).convert('RGB'),dtype=np.uint8)

        im = vis_transform(im)
        
        gt = label[0].data.cpu().numpy()
        gt = colorize_mask(gt).convert('RGB')
        gt = vis_transform(gt)

        pred = output[0].data.max(0)[1].cpu().numpy()
        pred = colorize_mask(pred).convert('RGB')
        pred = vis_transform(pred)

        grid = torch.stack([im,gt,pred],0)
        grid = make_grid(grid.cpu(), nrow=3, padding=5)
        
        writer.add_image(f'img_{np.array(index)[0]}', grid, step)
        print(index.numpy())
        break

if __name__ == "__main__":
    train()