"""
TAR loss is the self-supervision losss for the target domain based on pseudo=labels
Typically 
    Entropy minimization
    MasSquare

This file will use MaxSquare
L_TAR = -1/HW * sum(sum(p_t))

@Author: XIaofeng Zhang
@Date: 2022/9/21
"""
from turtle import forward
import torch
import torch.nn as nn


class MaxSquareLoss(nn.Module):
    def __init__(self,ignore_index = 255, num_class=19):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class

    def forward(self,prob):
        """
        :param prob: probability of pred (B,C,H,W)

        :return: maximum square loss
        """

        mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss