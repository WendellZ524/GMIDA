"""
MAE loss means mean absolute error

inputs:
Fake -> fixed   source -> segmentation result as ys~
Fake -> updated source -> segmentation result as S(xs~)



L_MAE = E{ L1[ S(xs~) - ys~ ] }

@Author: XIaofeng Zhang
@Date: 2022/9/21
"""

import torch
import numpy as np
import torch.nn as nn
class TARLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input,target):
        """
        :param prob: probability of pred (B,C,H,W)

        :return: maximum square loss
        """

        mae = nn.L1Loss(reduction = 'mean')
        loss = mae(input,target)
        
        return loss