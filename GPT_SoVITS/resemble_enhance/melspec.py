import numpy as np
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram as TorchMelSpectrogram

from .hparams import HParams


class MelSpectrogram(nn.Module):
    def __init__(self, hp: HParams):
        """
        Torch implementation of Resemble's mel extraction.
        Note that the values are NOT identical to librosa's implementation
        due to floating point precisions.
        """
        super().__init__()
        self.hp = hp
        self.melspec = TorchMelSpectrogram(
            hp.wav_rate,
            n_fft=hp.n_fft,
            win_length=hp.win_size,
            hop_length=hp.hop_size,
            f_min=0,
            f_max=hp.wav_rate // 2,
            n_mels=hp.num_mels,
            power=1,
            normalized=False,
            # NOTE: Folowing librosa's default.
            pad_mode="constant",
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("stft_magnitude_min", torch.FloatTensor([hp.stft_magnitude_min]))
        self.min_level_db = 20 * np.log10(hp.stft_magnitude_min)
        self.preemphasis = hp.preemphasis
        self.hop_size = hp.hop_size

    def forward(self, wav, pad=True):
        """
        Args:
            wav: [B, T]
        """
        device = wav.device
        if wav.is_mps:
            wav = wav.cpu()
            self.to(wav.device)
        if self.preemphasis > 0:
            wav = torch.nn.functional.pad(wav, [1, 0], value=0)
            wav = wav[..., 1:] - self.preemphasis * wav[..., :-1]
        mel = self.melspec(wav)
        mel = self._amp_to_db(mel)
        mel_normed = self._normalize(mel)
        assert not pad or mel_normed.shape[-1] == 1 + wav.shape[-1] // self.hop_size  # Sanity check
        mel_normed = mel_normed.to(device)
        return mel_normed  # (M, T)

    def _normalize(self, s, headroom_db=15):
        return (s - self.min_level_db) / (-self.min_level_db + headroom_db)

    def _amp_to_db(self, x):
        return x.clamp_min(self.hp.stft_magnitude_min).log10() * 20
