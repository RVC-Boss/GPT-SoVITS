# Refer from https://github.com/NVIDIA/BigVGAN

import math

import torch
import torch.nn as nn
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .alias_free_torch import DownSample1d, UpSample1d


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, clamp=(1e-2, 50)):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = in_features
        self.log_alpha = nn.Parameter(torch.zeros(in_features) + math.log(alpha))
        self.log_beta = nn.Parameter(torch.zeros(in_features) + math.log(alpha))
        self.clamp = clamp

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.log_alpha.exp().clamp(*self.clamp)
        alpha = alpha[None, :, None]

        beta = self.log_beta.exp().clamp(*self.clamp)
        beta = beta[None, :, None]

        x = x + (1.0 / beta) * (x * alpha).sin().pow(2)

        return x


class UpActDown(nn.Module):
    def __init__(
        self,
        act,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = act
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        # x: [B,C,T]
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


class AMPBlock(nn.Sequential):
    def __init__(self, channels, *, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__(*(self._make_layer(channels, kernel_size, d) for d in dilations))

    def _make_layer(self, channels, kernel_size, dilation):
        return nn.Sequential(
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding="same")),
            UpActDown(act=SnakeBeta(channels)),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, padding="same")),
        )

    def forward(self, x):
        return x + super().forward(x)
