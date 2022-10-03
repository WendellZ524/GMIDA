"""losses.py
Author: Xiaofen Zhang
Purpose: this file defines lossed used by SFDA 
References: 
SFDA: https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Source-Free_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.pdf
MAE loss
"""
import torch
import numpy as np
import torch.nn as nn
class TARLoss(nn.Module):
    """
    MAE loss means mean absolute error

    inputs:
    Fake -> fixed   source -> segmentation result as ys~
    Fake -> updated source -> segmentation result as S(xs~)



    L_MAE = E{ L1[ S(xs~) - ys~ ] }

    @Author: XIaofeng Zhang
    @Date: 2022/9/21
    """

    def __init__(self):
        super().__init__()

    def forward(self,input,target):
        """MAE loss

        Args:
            input (tensor): S( G(z) ) : fake image -> S() 
            target (tensor): S~( G(z) ) fake image -> S~() fixed source model

        Returns:
            mae_loss (tensor): Mean Absolute Error loss
        """

        mae = nn.L1Loss(reduction = 'mean')
        loss = mae(input,target)
        
        return loss