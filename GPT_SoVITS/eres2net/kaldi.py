"""Kaldi-compatible log-mel features used by the speaker inference model."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

_EPSILON = torch.tensor(torch.finfo(torch.float).eps)
_MEL_BANKS = {}


def _mel_banks(num_bins, padded_size, sample_frequency, device, dtype):
    key = (num_bins, padded_size, sample_frequency, device, dtype)
    if key not in _MEL_BANKS:
        assert num_bins > 3, "Must have at least 3 mel bins"
        low_mel = 1127.0 * math.log(1.0 + 20.0 / 700.0)
        high_mel = 1127.0 * math.log(1.0 + sample_frequency * 0.5 / 700.0)
        delta = (high_mel - low_mel) / (num_bins + 1)
        bins = torch.arange(num_bins).unsqueeze(1)
        left = low_mel + bins * delta
        center = low_mel + (bins + 1.0) * delta
        right = low_mel + (bins + 2.0) * delta
        frequencies = (sample_frequency / padded_size) * torch.arange(padded_size / 2)
        mel = (1127.0 * (1.0 + frequencies / 700.0).log()).unsqueeze(0)
        up = (mel - left) / (center - left)
        down = (right - mel) / (right - center)
        banks = torch.max(torch.zeros(1), torch.min(up, down))
        _MEL_BANKS[key] = banks.to(device=device, dtype=dtype)
    return _MEL_BANKS[key]


def fbank(waveform: Tensor, num_mel_bins=80, sample_frequency=16000.0, dither=0.0) -> Tensor:
    """Extract the 25 ms Povey-window log-mel features expected by ERes2NetV2."""
    assert waveform.dim() == 2 and waveform.size(0) > 0, "Expected audio of shape (channels, samples)"
    waveform = waveform[0, :]
    device, dtype = waveform.device, waveform.dtype
    window_shift = int(sample_frequency * 10.0 * 0.001)
    window_size = int(sample_frequency * 25.0 * 0.001)
    assert sample_frequency > 40, "sample_frequency must exceed the 20 Hz low cutoff"
    assert 2 <= window_size <= len(waveform), "Audio must contain at least one 25 ms frame"
    assert window_shift > 0, "window_shift must be greater than zero"
    padded_size = 2 ** (window_size - 1).bit_length()
    count = 1 + (waveform.size(0) - window_size) // window_shift
    frames = waveform.as_strided((count, window_size), (window_shift * waveform.stride(0), waveform.stride(0)))
    if dither != 0.0:
        frames = frames + torch.randn(frames.shape, device=device, dtype=dtype) * dither
    frames = frames - torch.mean(frames, dim=1).unsqueeze(1)
    previous = F.pad(frames.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
    frames = frames - 0.97 * previous[:, :-1]
    window = torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85)
    frames = frames * window.unsqueeze(0)
    if padded_size != window_size:
        frames = F.pad(frames.unsqueeze(0), (0, padded_size - window_size), mode="constant", value=0).squeeze(0)
    spectrum = torch.fft.rfft(frames).abs().pow(2.0)
    banks = _mel_banks(num_mel_bins, padded_size, sample_frequency, device, dtype)
    banks = F.pad(banks, (0, 1), mode="constant", value=0)
    energies = torch.mm(spectrum, banks.T)
    return torch.max(energies, _EPSILON.to(device=device, dtype=dtype)).log()
