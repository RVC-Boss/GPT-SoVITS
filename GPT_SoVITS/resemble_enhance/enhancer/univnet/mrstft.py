# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)


import torch
import torch.nn.functional as F
from torch import nn

from ..hparams import HParams


def _make_stft_cfg(hop_length, win_length=None):
    if win_length is None:
        win_length = 4 * hop_length
    n_fft = 2 ** (win_length - 1).bit_length()
    return dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def get_stft_cfgs(hp: HParams):
    assert hp.wav_rate == 44100, f"wav_rate must be 44100, got {hp.wav_rate}"
    return [_make_stft_cfg(h) for h in (100, 256, 512)]


def stft(x, n_fft, hop_length, win_length, window):
    dtype = x.dtype
    x = torch.stft(x.float(), n_fft, hop_length, win_length, window, return_complex=True)
    x = x.abs().to(dtype)
    x = x.transpose(2, 1)  # (b f t) -> (b t f)
    return x


class SpectralConvergengeLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log1p(x_mag), torch.log1p(y_mag))


class STFTLoss(nn.Module):
    def __init__(self, hp, stft_cfg: dict, window="hann_window"):
        super().__init__()
        self.hp = hp
        self.stft_cfg = stft_cfg
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.register_buffer("window", getattr(torch, window)(stft_cfg["win_length"]), persistent=False)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        stft_cfg = dict(self.stft_cfg)
        x_mag = stft(x, **stft_cfg, window=self.window)  # (b t) -> (b t f)
        y_mag = stft(y, **stft_cfg, window=self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return dict(sc=sc_loss, mag=mag_loss)


class MRSTFTLoss(nn.Module):
    def __init__(self, hp: HParams, window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            resolutions (list): List of (FFT size, hop size, window length).
            window (str): Window function type.
        """
        super().__init__()
        stft_cfgs = get_stft_cfgs(hp)
        self.stft_losses = nn.ModuleList()
        self.hp = hp
        for c in stft_cfgs:
            self.stft_losses += [STFTLoss(hp, c, window=window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (b t).
            y (Tensor): Groundtruth signal (b t).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        assert x.dim() == 2 and y.dim() == 2, f"(b t) is expected, but got {x.shape} and {y.shape}."

        dtype = x.dtype

        x = x.float()
        y = y.float()

        # Align length
        x = x[..., : y.shape[-1]]
        y = y[..., : x.shape[-1]]

        losses = {}

        for f in self.stft_losses:
            d = f(x, y)
            for k, v in d.items():
                losses.setdefault(k, []).append(v)

        for k, v in losses.items():
            losses[k] = torch.stack(v, dim=0).mean().to(dtype)

        return losses
