#!/usr/bin/env python3.8
import math

import torch.nn as nn


def maxpool() -> nn.MaxPool2d:
    """Create a 2x2 MaxPool2d layer with stride 2.

    Returns:
        nn.MaxPool2d: Max pooling layer.
    """
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


def conv_block(in_dim: int,
               out_dim: int,
               act_fn: nn.Module,
               kernel_size: int = 3,
               stride: int = 1,
               padding: int = 1,
               dilation: int = 1
               ) -> nn.Sequential:
    """Convolutional block: Conv2d → BatchNorm2d → Activation.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        act_fn (nn.Module): Activation function (e.g., ReLU).
        kernel_size (int): Convolution kernel size.
        stride (int): Convolution stride.
        padding (int): Padding size.
        dilation (int): Dilation factor.

    Returns:
        nn.Sequential: [Conv2d, BatchNorm2d, Activation]
    """
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  dilation=dilation),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )


def conv_block_3(in_dim: int,
                 out_dim: int,
                 act_fn: nn.Module
                 ) -> nn.Sequential:
    """Three-block sequence: two conv_blocks + Conv2d/BN.

    Structure:
      1) conv_block(in_dim → out_dim)
      2) conv_block(out_dim → out_dim)
      3) Conv2d(out_dim → out_dim) + BatchNorm

    Args:
        in_dim (int): Input channels for the first conv.
        out_dim (int): Output channels (fixed for all).
        act_fn (nn.Module): Activation for the first two blocks.

    Returns:
        nn.Sequential: [conv_block, conv_block, Conv2d, BatchNorm2d]
    """
    return nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )


def conv_decod_block(in_dim: int,
                     out_dim: int,
                     act_fn: nn.Module
                     ) -> nn.Sequential:
    """Decoder block: ConvTranspose2d → BatchNorm2d → Activation.

    Args:
        in_dim (int): Input channels.
        out_dim (int): Output channels.
        act_fn (nn.Module): Activation function (e.g., ReLU).

    Returns:
        nn.Sequential: [ConvTranspose2d, BatchNorm2d, Activation]
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )


class Conv_residual_conv(nn.Module):
    """Residual block: conv → 3xconv block → sum → conv."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 act_fn: nn.Module):
        """
        Args:
            in_dim (int): Input channels.
            out_dim (int): Output channels for all sub-blocks.
            act_fn (nn.Module): Activation for the conv_blocks.
        """
        super().__init__()
        self.conv_1 = conv_block(in_dim,  out_dim, act_fn)
        self.conv_2 = conv_block_3(out_dim, out_dim, act_fn)
        self.conv_3 = conv_block(out_dim,  out_dim, act_fn)

    def forward(self, x):
        # 1) First convolution
        c1 = self.conv_1(x)
        # 2) Three-layer conv block
        c2 = self.conv_2(c1)
        # 3) Residual sum
        res = c1 + c2
        # 4) Final convolution
        return self.conv_3(res)


class ResidualUNet(nn.Module):
    """U-Net with residual blocks and average skip connections."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int):
        """
        Args:
            in_dim (int): Input image channels.
            out_dim (int): Output channels/classes.
        """
        super().__init__()
        ngf = 32                        # base number of filters
        act_down = nn.LeakyReLU(0.2, True)
        act_up   = nn.ReLU()

        # Encoder
        self.down_1 = Conv_residual_conv(in_dim,      ngf,      act_down)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(ngf,         ngf * 2,  act_down)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(ngf * 2,     ngf * 4,  act_down)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(ngf * 4,     ngf * 8,  act_down)
        self.pool_4 = maxpool()

        # Bridge (deepest layer)
        self.bridge = Conv_residual_conv(ngf * 8,     ngf * 16, act_down)

        # Decoder and forward pass (not shown)
