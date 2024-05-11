import logging
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from ...common import Normalizer

logger = logging.getLogger(__name__)


@dataclass
class IRMAEOutput:
    latent: Tensor  # latent vector
    decoded: Tensor | None  # decoder output, include extra dim


class ResBlock(nn.Sequential):
    def __init__(self, channels, dilations=[1, 2, 4, 8]):
        wn = weight_norm
        super().__init__(
            nn.GroupNorm(32, channels),
            nn.GELU(),
            wn(nn.Conv1d(channels, channels, 3, padding="same", dilation=dilations[0])),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            wn(nn.Conv1d(channels, channels, 3, padding="same", dilation=dilations[1])),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            wn(nn.Conv1d(channels, channels, 3, padding="same", dilation=dilations[2])),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            wn(nn.Conv1d(channels, channels, 3, padding="same", dilation=dilations[3])),
        )

    def forward(self, x: Tensor):
        return x + super().forward(x)


class IRMAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim,
        hidden_dim=1024,
        num_irms=4,
    ):
        """
        Args:
            input_dim: input dimension
            output_dim: output dimension
            latent_dim: latent dimension
            hidden_dim: hidden layer dimension
            num_irm_matrics: number of implicit rank minimization matrices
            norm: normalization layer
        """
        self.input_dim = input_dim
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding="same"),
            *[ResBlock(hidden_dim) for _ in range(4)],
            # Try to obtain compact representation (https://proceedings.neurips.cc/paper/2020/file/a9078e8653368c9c291ae2f8b74012e7-Paper.pdf)
            *[nn.Conv1d(hidden_dim if i == 0 else latent_dim, latent_dim, 1, bias=False) for i in range(num_irms)],
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, 3, padding="same"),
            *[ResBlock(hidden_dim) for _ in range(4)],
            nn.Conv1d(hidden_dim, output_dim, 1),
        )

        self.head = nn.Sequential(
            nn.Conv1d(output_dim, hidden_dim, 3, padding="same"),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, 1),
        )

        self.estimator = Normalizer()

    def encode(self, x):
        """
        Args:
            x: (b c t) tensor
        """
        z = self.encoder(x)  # (b c t)
        _ = self.estimator(z)  # Estimate the glboal mean and std of z
        self.stats = {}
        self.stats["z_mean"] = z.mean().item()
        self.stats["z_std"] = z.std().item()
        self.stats["z_abs_68"] = z.abs().quantile(0.6827).item()
        self.stats["z_abs_95"] = z.abs().quantile(0.9545).item()
        self.stats["z_abs_99"] = z.abs().quantile(0.9973).item()
        return z

    def decode(self, z):
        """
        Args:
            z: (b c t) tensor
        """
        return self.decoder(z)

    def forward(self, x, skip_decoding=False):
        """
        Args:
            x: (b c t) tensor
            skip_decoding: if True, skip the decoding step
        """
        z = self.encode(x)  # q(z|x)

        if skip_decoding:
            # This speeds up the training in cfm only mode
            decoded = None
        else:
            decoded = self.decode(z)  # p(x|z)
            predicted = self.head(decoded)
            self.losses = dict(mse=F.mse_loss(predicted, x))

        return IRMAEOutput(latent=z, decoded=decoded)
