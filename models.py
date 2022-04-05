# CMU 16-726 Learning-Based Image Synthesis / Spring 2022, Assignment 3
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - CycleGenerator     --> Used in the CycleGAN in Part 2
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
#   - PatchDiscriminator --> Used in the CycleGAN in Part 2
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ and forward methods in the
# DCGenerator, CycleGenerator, DCDiscriminator, and PatchDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1, scale_factor=2, norm='batch'):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    if norm == 'spectral':
        layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='batch', init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    if norm == 'spectral':
        conv_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False))
    else:
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))
    return nn.Sequential(*layers)




class ResnetBlock(nn.Module):
    def __init__(self, conv_dim, norm):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False, norm='batch'):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        # in: 3 * 64 * 64
        self.conv1 = conv(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, norm='instance')
        # ->  32 * 32 * 32
        self.conv2 = conv(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, norm='instance')
        # ->  64 * 16 * 16

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim=64, norm=norm)
        # ->  64 * 16 * 16

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.up_conv1 = up_conv(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='instance')
        # ->  32 * 32 * 32
        self.up_conv2 = up_conv(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='none')
        # ->   3 * 64 * 64

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        x = self.conv1(x)
        x = self.relu(x)
        assert(x.shape[1:] == torch.Size([32, 32, 32]), x.shape[1:])
        # print(x.shape[1:], 'G_conv1')

        x = self.conv2(x)
        x = self.relu(x)
        assert(x.shape[1:] == torch.Size([64, 16, 16]), x.shape[1:])
        # print(x.shape[1:], 'G_conv2')

        x = self.resnet_block(x)
        x = self.resnet_block(x)
        x = self.resnet_block(x)
        x = self.resnet_block(x)
        x = self.resnet_block(x)
        x = self.resnet_block(x)
        x = self.relu(x)
        # print(x.shape[1:], 'G_conv3')

        x = self.up_conv1(x)
        x = self.relu(x)
        # print(x.shape[1:], 'G_conv4')

        x = self.up_conv2(x)
        x = self.tanh(x)
        # print(x.shape[1:], 'G_conv5')

        return x




class PatchDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, norm='batch'):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # Hint: it should look really similar to DCDiscriminator.
        # in: 3 * 64 * 64
        self.conv1 = conv(in_channels=3, out_channels=64, padding=1, stride=2, kernel_size=4, norm='instance')
        # -> 32 * 32 * 32
        self.conv2 = conv(in_channels=64, out_channels=128, padding=1, stride=2, kernel_size=4, norm='instance')
        # -> 64 * 16 * 16
        self.conv3 = conv(in_channels=128, out_channels=256, padding=1, stride=2, kernel_size=4, norm='instance')
        # -> 128 * 8 * 8
        self.conv4 = conv(in_channels=256, out_channels=1, padding=1, stride=2, kernel_size=4, norm='none')
        # -> 1 * 4 * 4

        self.relu = nn.ReLU()

    def forward(self, x):

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        x = self.conv1(x)
        x = self.relu(x)
        # assert(x.shape[1:] == torch.Size([32, 32, 32]) and "d_conv1")

        x = self.conv2(x)
        x = self.relu(x)
        # assert(x.shape[1:] == torch.Size([64, 16, 16]) and "d_conv2")

        x = self.conv3(x)
        x = self.relu(x)
        # assert(x.shape[1:] == torch.Size([128, 8, 8]) and "d_conv3")

        x = self.conv4(x)
        # assert(x.shape[1:] == torch.Size([1, 4, 4]) and "d_conv4")

        return x
