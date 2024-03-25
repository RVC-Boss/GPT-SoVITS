import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as DatasetBase

from ..hparams import HParams
from .distorter import Distorter
from .utils import rglob_audio_files

logger = logging.getLogger(__name__)


def _normalize(x):
    return x / (np.abs(x).max() + 1e-7)


def _collate(batch, key, tensor=True, pad=True):
    l = [d[key] for d in batch]
    if l[0] is None:
        return None
    if tensor:
        l = [torch.from_numpy(x) for x in l]
    if pad:
        assert tensor, "Can't pad non-tensor"
        l = pad_sequence(l, batch_first=True)
    return l


def praat_augment(wav, sr):
    try:
        import parselmouth
    except ImportError:
        raise ImportError("Please install parselmouth>=0.5.0 to use Praat augmentation")
    # "praat-parselmouth @ git+https://github.com/YannickJadoul/Parselmouth@0bbcca69705ed73322f3712b19d71bb3694b2540",
    # https://github.com/YannickJadoul/Parselmouth/issues/68
    # note that this function may hang if the praat version is 0.4.3
    assert wav.ndim == 1, f"wav.ndim must be 1 but got {wav.ndim}"
    sound = parselmouth.Sound(wav, sr)
    formant_shift_ratio = random.uniform(1.1, 1.5)
    pitch_range_factor = random.uniform(0.5, 2.0)
    sound = parselmouth.praat.call(sound, "Change gender", 75, 600, formant_shift_ratio, 0, pitch_range_factor, 1.0)
    wav = np.array(sound.values)[0].astype(np.float32)
    return wav


class Dataset(DatasetBase):
    def __init__(
        self,
        fg_paths: list[Path],
        hp: HParams,
        training=True,
        max_retries=100,
        silent_fg_prob=0.01,
        mode=False,
    ):
        super().__init__()

        assert mode in ("enhancer", "denoiser"), f"Invalid mode: {mode}"

        self.hp = hp
        self.fg_paths = fg_paths
        self.bg_paths = rglob_audio_files(hp.bg_dir)

        if len(self.fg_paths) == 0:
            raise ValueError(f"No foreground audio files found in {hp.fg_dir}")

        if len(self.bg_paths) == 0:
            raise ValueError(f"No background audio files found in {hp.bg_dir}")

        logger.info(f"Found {len(self.fg_paths)} foreground files and {len(self.bg_paths)} background files")

        self.training = training
        self.max_retries = max_retries
        self.silent_fg_prob = silent_fg_prob

        self.mode = mode
        self.distorter = Distorter(hp, training=training, mode=mode)

    def _load_wav(self, path, length=None, random_crop=True):
        wav, sr = torchaudio.load(path)

        wav = AF.resample(
            waveform=wav,
            orig_freq=sr,
            new_freq=self.hp.wav_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )

        wav = wav.float().numpy()

        if wav.ndim == 2:
            wav = np.mean(wav, axis=0)

        if length is None and self.training:
            length = int(self.hp.training_seconds * self.hp.wav_rate)

        if length is not None:
            if random_crop:
                start = random.randint(0, max(0, len(wav) - length))
                wav = wav[start : start + length]
            else:
                wav = wav[:length]

        if length is not None and len(wav) < length:
            wav = np.pad(wav, (0, length - len(wav)))

        wav = _normalize(wav)

        return wav

    def _getitem_unsafe(self, index: int):
        fg_path = self.fg_paths[index]

        if self.training and random.random() < self.silent_fg_prob:
            fg_wav = np.zeros(int(self.hp.training_seconds * self.hp.wav_rate), dtype=np.float32)
        else:
            fg_wav = self._load_wav(fg_path)
            if random.random() < self.hp.praat_augment_prob and self.training:
                fg_wav = praat_augment(fg_wav, self.hp.wav_rate)

        if self.hp.load_fg_only:
            bg_wav = None
            fg_dwav = None
            bg_dwav = None
        else:
            fg_dwav = _normalize(self.distorter(fg_wav, self.hp.wav_rate)).astype(np.float32)
            if self.training:
                bg_path = random.choice(self.bg_paths)
            else:
                # Deterministic for validation
                bg_path = self.bg_paths[index % len(self.bg_paths)]
            bg_wav = self._load_wav(bg_path, length=len(fg_wav), random_crop=self.training)
            bg_dwav = _normalize(self.distorter(bg_wav, self.hp.wav_rate)).astype(np.float32)

        return dict(
            fg_wav=fg_wav,
            bg_wav=bg_wav,
            fg_dwav=fg_dwav,
            bg_dwav=bg_dwav,
        )

    def __getitem__(self, index: int):
        for i in range(self.max_retries):
            try:
                return self._getitem_unsafe(index)
            except Exception as e:
                if i == self.max_retries - 1:
                    raise RuntimeError(f"Failed to load {self.fg_paths[index]} after {self.max_retries} retries") from e
                logger.debug(f"Error loading {self.fg_paths[index]}: {e}, skipping")
                index = np.random.randint(0, len(self))

    def __len__(self):
        return len(self.fg_paths)

    @staticmethod
    def collate_fn(batch):
        return dict(
            fg_wavs=_collate(batch, "fg_wav"),
            bg_wavs=_collate(batch, "bg_wav"),
            fg_dwavs=_collate(batch, "fg_dwav"),
            bg_dwavs=_collate(batch, "bg_dwav"),
        )
