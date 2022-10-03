"""losses.py
Author: Xiaofen Zhang
Purpose: this file defines lossed used by SFDA 

losses for segmentation:
focal loss

including losses for SFDA: 
SFDA: https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Source-Free_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.pdf
MAE loss:

"""
import torch.nn.functional as F
import torch
import torch.nn as nn


from turtle import forward
import torch
import torch.nn as nn




class TARLoss(nn.Module):
    """
    TAR loss is the self-supervision losss for the target domain based on pseudo=labels
    Typically 
        Entropy minimization
        MasSquare

    This file will use MaxSquare
    L_TAR = -1/HW * sum(sum(p_t))

    @Author: XIaofeng Zhang
    @Date: 2022/9/21
    @Source: https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Domain_Adaptation_for_Semantic_Segmentation_With_Maximum_Squares_Loss_ICCV_2019_paper.pdf
    @reference: https://github.com/ZJULearning/MaxSquareLoss
    """
    def __init__(self,ignore_index = 255, num_class=19):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class

    def forward(self,prob):
        """Tar loss

        Args:
            prob (tensor): probability of pred (B,C,H,W) C is n_classes

        Returns:
            tar loss (tensor) : MaxSquare tar loss
        """
        mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss

class MAELoss(nn.Module):
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

######### segmentation losses #############
class FocalLoss(nn.Module):
    """https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/utils/loss.py

    Args:
        nn (_type_): _description_
    """
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()