import torch
import torch.nn as nn

# Implementing DCGAN (Deep Convolutional GAN): Details in the original paper

class Discriminator(nn.Module):
    """
    Discriminator for DCGAN: Specifications as per original paper
    """

    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, 2 * features_d, kernel_size=4, stride=2, padding=1),  # 16x16
            self._block(2 * features_d, 4 * features_d, kernel_size=4, stride=2, padding=1),  # 8x8
            self._block(4 * features_d, 8 * features_d, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.Conv2d(8 * features_d, 1, kernel_size=4, stride=2, padding=0),  # 1x1
            nn.Sigmoid()  # Tells probability of being real/fake
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False  # Due to BatchNorm
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

### Initializing the Weights as per the paper:
### Weights are initialized from a zero-mean normal deviation with std-dev = 0.02

def initialize_weights(model):
    for m in model.modules():
        # Note: No fully connected layers are present in Generator/Discriminator
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Testing the model
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    # Discriminator test
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc),
    x = torch.randn((N, in_channels, H, W))
    assert disc(x).shape == (N, 1, 1, 1)

    # Generator Test
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape  == (N, in_channels, H, W)

    print("All tests passed!")

# test()