import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from ..hparams import HParams
from .lvcnet import LVCBlock
from .mrstft import MRSTFTLoss


class UnivNet(nn.Module):
    @property
    def d_noise(self):
        return 128

    @property
    def strides(self):
        return [7, 5, 4, 3]

    @property
    def dilations(self):
        return [1, 3, 9, 27]

    @property
    def nc(self):
        return self.hp.univnet_nc

    @property
    def scale_factor(self) -> int:
        return self.hp.hop_size

    def __init__(self, hp: HParams, d_input):
        super().__init__()
        self.d_input = d_input

        self.hp = hp

        self.blocks = nn.ModuleList(
            [
                LVCBlock(
                    self.nc,
                    d_input,
                    stride=stride,
                    dilations=self.dilations,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=3,
                )
                for stride, hop_length in zip(self.strides, np.cumprod(self.strides))
            ]
        )

        self.conv_pre = weight_norm(nn.Conv1d(self.d_noise, self.nc, 7, padding=3, padding_mode="reflect"))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(self.nc, 1, 7, padding=3, padding_mode="reflect")),
            nn.Tanh(),
        )

        self.mrstft = MRSTFTLoss(hp)

    @property
    def eps(self):
        return 1e-5

    def forward(self, x: Tensor, y: Tensor | None = None, npad=10):
        """
        Args:
            x: (b c t), acoustic features
            y: (b t), waveform
        Returns:
            z: (b t), waveform
        """
        assert x.ndim == 3, "x must be 3D tensor"
        assert y is None or y.ndim == 2, "y must be 2D tensor"
        assert x.shape[1] == self.d_input, f"x.shape[1] must be {self.d_input}, but got {x.shape}"
        assert npad >= 0, "npad must be positive or zero"

        x = F.pad(x, (0, npad), "constant", 0)
        z = torch.randn(x.shape[0], self.d_noise, x.shape[2]).to(x)
        z = self.conv_pre(z)  # (b c t)

        for block in self.blocks:
            z = block(z, x)  # (b c t)

        z = self.conv_post(z)  # (b 1 t)
        z = z[..., : -self.scale_factor * npad]
        z = z.squeeze(1)  # (b t)

        if y is not None:
            self.losses = self.mrstft(z, y)

        return z
