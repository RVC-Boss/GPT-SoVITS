# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import torch
import torch.nn as nn
from alias_free_activation.torch.resample import UpSample1d, DownSample1d

# load fused CUDA kernel: this enables importing anti_alias_activation_cuda
from alias_free_activation.cuda import load

anti_alias_activation_cuda = load.load()


class FusedAntiAliasActivation(torch.autograd.Function):
    """
    Assumes filter size 12, replication padding on upsampling/downsampling, and logscale alpha/beta parameters as inputs.
    The hyperparameters are hard-coded in the kernel to maximize speed.
    NOTE: The fused kenrel is incorrect for Activation1d with different hyperparameters.
    """

    @staticmethod
    def forward(ctx, inputs, up_ftr, down_ftr, alpha, beta):
        activation_results = anti_alias_activation_cuda.forward(
            inputs, up_ftr, down_ftr, alpha, beta
        )

        return activation_results

    @staticmethod
    def backward(ctx, output_grads):
        raise NotImplementedError
        return output_grads, None, None


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        fused: bool = True,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

        self.fused = fused  # Whether to use fused CUDA kernel or not

    def forward(self, x):
        if not self.fused:
            x = self.upsample(x)
            x = self.act(x)
            x = self.downsample(x)
            return x
        else:
            if self.act.__class__.__name__ == "Snake":
                beta = self.act.alpha.data  # Snake uses same params for alpha and beta
            else:
                beta = (
                    self.act.beta.data
                )  # Snakebeta uses different params for alpha and beta
            alpha = self.act.alpha.data
            if (
                not self.act.alpha_logscale
            ):  # Exp baked into cuda kernel, cancel it out with a log
                alpha = torch.log(alpha)
                beta = torch.log(beta)

            x = FusedAntiAliasActivation.apply(
                x, self.upsample.filter, self.downsample.lowpass.filter, alpha, beta
            )
            return x
