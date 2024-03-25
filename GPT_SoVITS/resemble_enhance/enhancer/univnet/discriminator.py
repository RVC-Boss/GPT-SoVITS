import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from ..hparams import HParams
from .mrstft import get_stft_cfgs

logger = logging.getLogger(__name__)


class PeriodNetwork(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        wn = weight_norm
        self.convs = nn.ModuleList(
            [
                wn(nn.Conv2d(1, 64, (5, 1), (3, 1), padding=(2, 0))),
                wn(nn.Conv2d(64, 128, (5, 1), (3, 1), padding=(2, 0))),
                wn(nn.Conv2d(128, 256, (5, 1), (3, 1), padding=(2, 0))),
                wn(nn.Conv2d(256, 512, (5, 1), (3, 1), padding=(2, 0))),
                wn(nn.Conv2d(512, 1024, (5, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = wn(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T]
        """
        assert x.dim() == 3, f"(B, 1, T) is expected, but got {x.shape}."

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x


class SpecNetwork(nn.Module):
    def __init__(self, stft_cfg: dict):
        super().__init__()
        wn = weight_norm
        self.stft_cfg = stft_cfg
        self.convs = nn.ModuleList(
            [
                wn(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                wn(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                wn(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                wn(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                wn(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = wn(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T]
        """
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
        x = self.conv_post(x)
        x = x.flatten(1, -1)
        return x

    def spectrogram(self, x):
        """
        Args:
            x: [B, 1, T]
        """
        x = x.squeeze(1)
        dtype = x.dtype
        stft_cfg = dict(self.stft_cfg)
        x = torch.stft(x.float(), center=False, return_complex=False, **stft_cfg)
        mag = x.norm(p=2, dim=-1)  # [B, F, TT]
        mag = mag.to(dtype)  # [B, F, TT]
        return mag


class MD(nn.ModuleList):
    def __init__(self, l: list):
        super().__init__([self._create_network(x) for x in l])
        self._loss_type = None

    def loss_type_(self, loss_type):
        self._loss_type = loss_type

    def _create_network(self, _):
        raise NotImplementedError

    def _forward_each(self, d, x, y):
        assert self._loss_type is not None, "loss_type is not set."
        loss_type = self._loss_type

        if loss_type == "hinge":
            if y == 0:
                # d(x) should be small -> -1
                loss = F.relu(1 + d(x)).mean()
            elif y == 1:
                # d(x) should be large -> 1
                loss = F.relu(1 - d(x)).mean()
            else:
                raise ValueError(f"Invalid y: {y}")
        elif loss_type == "wgan":
            if y == 0:
                loss = d(x).mean()
            elif y == 1:
                loss = -d(x).mean()
            else:
                raise ValueError(f"Invalid y: {y}")
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")

        return loss

    def forward(self, x, y) -> Tensor:
        losses = [self._forward_each(d, x, y) for d in self]
        return torch.stack(losses).mean()


class MPD(MD):
    def __init__(self):
        super().__init__([2, 3, 7, 13, 17])

    def _create_network(self, period):
        return PeriodNetwork(period)


class MRD(MD):
    def __init__(self, stft_cfgs):
        super().__init__(stft_cfgs)

    def _create_network(self, stft_cfg):
        return SpecNetwork(stft_cfg)


class Discriminator(nn.Module):
    @property
    def wav_rate(self):
        return self.hp.wav_rate

    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.stft_cfgs = get_stft_cfgs(hp)
        self.mpd = MPD()
        self.mrd = MRD(self.stft_cfgs)
        self.dummy_float: Tensor
        self.register_buffer("dummy_float", torch.zeros(0), persistent=False)

    def loss_type_(self, loss_type):
        self.mpd.loss_type_(loss_type)
        self.mrd.loss_type_(loss_type)

    def forward(self, fake, real=None):
        """
        Args:
            fake: [B T]
            real: [B T]
        """
        fake = fake.to(self.dummy_float)

        if real is None:
            self.loss_type_("wgan")
        else:
            length_difference = (fake.shape[-1] - real.shape[-1]) / real.shape[-1]
            assert length_difference < 0.05, f"length_difference should be smaller than 5%"

            self.loss_type_("hinge")
            real = real.to(self.dummy_float)

            fake = fake[..., : real.shape[-1]]
            real = real[..., : fake.shape[-1]]

        losses = {}

        assert fake.dim() == 2, f"(B, T) is expected, but got {fake.shape}."
        assert real is None or real.dim() == 2, f"(B, T) is expected, but got {real.shape}."

        fake = fake.unsqueeze(1)

        if real is None:
            losses["mpd"] = self.mpd(fake, 1)
            losses["mrd"] = self.mrd(fake, 1)
        else:
            real = real.unsqueeze(1)
            losses["mpd_fake"] = self.mpd(fake, 0)
            losses["mpd_real"] = self.mpd(real, 1)
            losses["mrd_fake"] = self.mrd(fake, 0)
            losses["mrd_real"] = self.mrd(real, 1)

        return losses
