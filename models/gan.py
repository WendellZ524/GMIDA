import torch
import torch.nn as nn
import torchvision

def dconv_bn_relu(in_dim, out_dim):
    """_summary_

    Args:
        in_dim (int): input channel
        out_dim (int): output channel

    Returns:
        nn.Model: Sequential model ConvTranspos2d -> BN -> Relu()

    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size = 5, stride = 2,
                                   padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU())

class Generator(nn.Module):
    def __init__(self, in_dim=100, dim=32):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 16 * 32, bias=False),
            nn.BatchNorm1d(dim * 8 * 16 * 32),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())


    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 16, 32)
        y = self.l2_5(y)
        return y

class Discriminator(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(Discriminator, self).__init__()
        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))
    
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y
