import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..melspec import MelSpectrogram
from .hparams import HParams
from .unet import UNet

logger = logging.getLogger(__name__)


def _normalize(x: Tensor) -> Tensor:
    return x / (x.abs().max(dim=-1, keepdim=True).values + 1e-7)


class Denoiser(nn.Module):
    @property
    def stft_cfg(self) -> dict:
        hop_size = self.hp.hop_size
        return dict(hop_length=hop_size, n_fft=hop_size * 4, win_length=hop_size * 4)

    @property
    def n_fft(self):
        return self.stft_cfg["n_fft"]

    @property
    def eps(self):
        return 1e-7

    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.net = UNet(input_dim=3, output_dim=3)
        self.mel_fn = MelSpectrogram(hp)

        self.dummy: Tensor
        self.register_buffer("dummy", torch.zeros(1), persistent=False)

    def to_mel(self, x: Tensor, drop_last=True):
        """
        Args:
            x: (b t), wavs
        Returns:
            o: (b c t), mels
        """
        if drop_last:
            return self.mel_fn(x)[..., :-1]  # (b d t)
        return self.mel_fn(x)

    def _stft(self, x):
        """
        Args:
            x: (b t)
        Returns:
            mag: (b f t) in [0, inf)
            cos: (b f t) in [-1, 1]
            sin: (b f t) in [-1, 1]
        """
        dtype = x.dtype
        device = x.device

        if x.is_mps:
            x = x.cpu()

        window = torch.hann_window(self.stft_cfg["win_length"], device=x.device)
        s = torch.stft(x.float(), **self.stft_cfg, window=window, return_complex=True)  # (b f t+1)

        s = s[..., :-1]  # (b f t)

        mag = s.abs()  # (b f t)

        phi = s.angle()  # (b f t)
        cos = phi.cos()  # (b f t)
        sin = phi.sin()  # (b f t)

        mag = mag.to(dtype=dtype, device=device)
        cos = cos.to(dtype=dtype, device=device)
        sin = sin.to(dtype=dtype, device=device)

        return mag, cos, sin

    def _istft(self, mag: Tensor, cos: Tensor, sin: Tensor):
        """
        Args:
            mag: (b f t) in [0, inf)
            cos: (b f t) in [-1, 1]
            sin: (b f t) in [-1, 1]
        Returns:
            x: (b t)
        """
        device = mag.device
        dtype = mag.dtype

        if mag.is_mps:
            mag = mag.cpu()
            cos = cos.cpu()
            sin = sin.cpu()

        real = mag * cos  # (b f t)
        imag = mag * sin  # (b f t)

        s = torch.complex(real, imag)  # (b f t)

        if s.isnan().any():
            logger.warning("NaN detected in ISTFT input.")

        s = F.pad(s, (0, 1), "replicate")  # (b f t+1)

        window = torch.hann_window(self.stft_cfg["win_length"], device=s.device)
        x = torch.istft(s, **self.stft_cfg, window=window, return_complex=False)

        if x.isnan().any():
            logger.warning("NaN detected in ISTFT output, set to zero.")
            x = torch.where(x.isnan(), torch.zeros_like(x), x)

        x = x.to(dtype=dtype, device=device)

        return x

    def _magphase(self, real, imag):
        mag = (real.pow(2) + imag.pow(2) + self.eps).sqrt()
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def _predict(self, mag: Tensor, cos: Tensor, sin: Tensor):
        """
        Args:
            mag: (b f t)
            cos: (b f t)
            sin: (b f t)
        Returns:
            mag_mask: (b f t) in [0, 1], magnitude mask
            cos_res: (b f t) in [-1, 1], phase residual
            sin_res: (b f t) in [-1, 1], phase residual
        """
        x = torch.stack([mag, cos, sin], dim=1)  # (b 3 f t)
        mag_mask, real, imag = self.net(x).unbind(1)  # (b 3 f t)
        mag_mask = mag_mask.sigmoid()  # (b f t)
        real = real.tanh()  # (b f t)
        imag = imag.tanh()  # (b f t)
        _, cos_res, sin_res = self._magphase(real, imag)  # (b f t)
        return mag_mask, sin_res, cos_res

    def _separate(self, mag, cos, sin, mag_mask, cos_res, sin_res):
        """Ref: https://audio-agi.github.io/Separate-Anything-You-Describe/AudioSep_arXiv.pdf"""
        sep_mag = F.relu(mag * mag_mask)
        sep_cos = cos * cos_res - sin * sin_res
        sep_sin = sin * cos_res + cos * sin_res
        return sep_mag, sep_cos, sep_sin

    def forward(self, x: Tensor, y: Tensor | None = None):
        """
        Args:
            x: (b t), a mixed audio
            y: (b t), a fg audio
        """
        assert x.dim() == 2, f"Expected (b t), got {x.size()}"
        x = x.to(self.dummy)
        x = _normalize(x)

        if y is not None:
            assert y.dim() == 2, f"Expected (b t), got {y.size()}"
            y = y.to(self.dummy)
            y = _normalize(y)

        mag, cos, sin = self._stft(x)  # (b 2f t)
        mag_mask, sin_res, cos_res = self._predict(mag, cos, sin)
        sep_mag, sep_cos, sep_sin = self._separate(mag, cos, sin, mag_mask, cos_res, sin_res)

        o = self._istft(sep_mag, sep_cos, sep_sin)

        npad = x.shape[-1] - o.shape[-1]
        o = F.pad(o, (0, npad))

        if y is not None:
            self.losses = dict(l1=F.l1_loss(o, y))

        return o
