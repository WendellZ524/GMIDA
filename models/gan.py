from math import fabs
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

class Generator(torch.nn.Module):
    def __init__(self, channels,z_dim = 100):
        super().__init__()
        k = 5 # 4 5
        p = 2 # 1 2
        op = 1 # 0 1
        self.z_dim=z_dim
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=k, stride=2, padding=p,output_padding=op),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=k, stride=2, padding=p,output_padding=op),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=k, stride=2, padding=p, output_padding=op))

            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Discriminator(torch.nn.Module):
    def __init__(self, channels=3,z_dim = 100):
        """
        Args:
            channels (int): channel of the input image
        Return:
            output (tensor): B 1 H/8 W/8 (128,256)->(16,32) 
        """
        super().__init__()
        # Filters [256, 512, 1024]
        self.z_dim = z_dim #(Cx64x64)
        k = 5 # 4 5
        p = 2 # 1 2
        op = 1 # 0 1
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=k, stride=2, padding=p),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=k, stride=2, padding=p),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=k, stride=2, padding=p),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=k, stride=1, padding=p))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.vie


if __name__ == "__main__":
    x = torch.randn(2,100,16,32) # 2x100
    G = Generator(3) # output channels
    fake = G(x) # 2,3, 32 ,32
    print(fake.size())
    D = Discriminator(3)
    r_logit = D(fake)
    print(r_logit.size())


