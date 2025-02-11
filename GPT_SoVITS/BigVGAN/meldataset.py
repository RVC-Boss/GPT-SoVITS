# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm
from typing import List, Tuple, Optional
from env import AttrDict

MAX_WAV_VALUE = 32767.0  # NOTE: 32768.0 -1 to prevent int16 overflow (results in popping sound in corner cases)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


mel_basis_cache = {}
hann_window_cache = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


def get_mel_spectrogram(wav, h):
    """
    Generate mel spectrogram from a waveform using given hyperparameters.

    Args:
        wav (torch.Tensor): Input waveform.
        h: Hyperparameters object with attributes n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    return mel_spectrogram(
        wav,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax,
    )


def get_dataset_filelist(a):
    training_files = []
    validation_files = []
    list_unseen_validation_files = []

    with open(a.input_training_file, "r", encoding="utf-8") as fi:
        training_files = [
            os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav")
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
        print(f"first training file: {training_files[0]}")

    with open(a.input_validation_file, "r", encoding="utf-8") as fi:
        validation_files = [
            os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav")
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
        print(f"first validation file: {validation_files[0]}")

    for i in range(len(a.list_input_unseen_validation_file)):
        with open(a.list_input_unseen_validation_file[i], "r", encoding="utf-8") as fi:
            unseen_validation_files = [
                os.path.join(a.list_input_unseen_wavs_dir[i], x.split("|")[0] + ".wav")
                for x in fi.read().split("\n")
                if len(x) > 0
            ]
            print(
                f"first unseen {i}th validation fileset: {unseen_validation_files[0]}"
            )
            list_unseen_validation_files.append(unseen_validation_files)

    return training_files, validation_files, list_unseen_validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files: List[str],
        hparams: AttrDict,
        segment_size: int,
        n_fft: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        sampling_rate: int,
        fmin: int,
        fmax: Optional[int],
        split: bool = True,
        shuffle: bool = True,
        device: str = None,
        fmax_loss: Optional[int] = None,
        fine_tuning: bool = False,
        base_mels_path: str = None,
        is_seen: bool = True,
    ):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.hparams = hparams
        self.is_seen = is_seen
        if self.is_seen:
            self.name = pathlib.Path(self.audio_files[0]).parts[0]
        else:
            self.name = "-".join(pathlib.Path(self.audio_files[0]).parts[:2]).strip("/")

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

        print("[INFO] checking dataset integrity...")
        for i in tqdm(range(len(self.audio_files))):
            assert os.path.exists(
                self.audio_files[i]
            ), f"{self.audio_files[i]} not found"

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        try:
            filename = self.audio_files[index]

            # Use librosa.load that ensures loading waveform into mono with [-1, 1] float values
            # Audio is ndarray with shape [T_time]. Disable auto-resampling here to minimize overhead
            # The on-the-fly resampling during training will be done only for the obtained random chunk
            audio, source_sampling_rate = librosa.load(filename, sr=None, mono=True)

            # Main logic that uses <mel, audio> pair for training BigVGAN
            if not self.fine_tuning:
                if self.split:  # Training step
                    # Obtain randomized audio chunk
                    if source_sampling_rate != self.sampling_rate:
                        # Adjust segment size to crop if the source sr is different
                        target_segment_size = math.ceil(
                            self.segment_size
                            * (source_sampling_rate / self.sampling_rate)
                        )
                    else:
                        target_segment_size = self.segment_size

                    # Compute upper bound index for the random chunk
                    random_chunk_upper_bound = max(
                        0, audio.shape[0] - target_segment_size
                    )

                    # Crop or pad audio to obtain random chunk with target_segment_size
                    if audio.shape[0] >= target_segment_size:
                        audio_start = random.randint(0, random_chunk_upper_bound)
                        audio = audio[audio_start : audio_start + target_segment_size]
                    else:
                        audio = np.pad(
                            audio,
                            (0, target_segment_size - audio.shape[0]),
                            mode="constant",
                        )

                    # Resample audio chunk to self.sampling rate
                    if source_sampling_rate != self.sampling_rate:
                        audio = librosa.resample(
                            audio,
                            orig_sr=source_sampling_rate,
                            target_sr=self.sampling_rate,
                        )
                        if audio.shape[0] > self.segment_size:
                            # trim last elements to match self.segment_size (e.g., 16385 for 44khz downsampled to 24khz -> 16384)
                            audio = audio[: self.segment_size]

                else:  # Validation step
                    # Resample full audio clip to target sampling rate
                    if source_sampling_rate != self.sampling_rate:
                        audio = librosa.resample(
                            audio,
                            orig_sr=source_sampling_rate,
                            target_sr=self.sampling_rate,
                        )
                    # Trim last elements to match audio length to self.hop_size * n for evaluation
                    if (audio.shape[0] % self.hop_size) != 0:
                        audio = audio[: -(audio.shape[0] % self.hop_size)]

                # BigVGAN is trained using volume-normalized waveform
                audio = librosa.util.normalize(audio) * 0.95

                # Cast ndarray to torch tensor
                audio = torch.FloatTensor(audio)
                audio = audio.unsqueeze(0)  # [B(1), self.segment_size]

                # Compute mel spectrogram corresponding to audio
                mel = mel_spectrogram(
                    audio,
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )  # [B(1), self.num_mels, self.segment_size // self.hop_size]

            # Fine-tuning logic that uses pre-computed mel. Example: Using TTS model-generated mel as input
            else:
                # For fine-tuning, assert that the waveform is in the defined sampling_rate
                # Fine-tuning won't support on-the-fly resampling to be fool-proof (the dataset should have been prepared properly)
                assert (
                    source_sampling_rate == self.sampling_rate
                ), f"For fine_tuning, waveform must be in the spcified sampling rate {self.sampling_rate}, got {source_sampling_rate}"

                # Cast ndarray to torch tensor
                audio = torch.FloatTensor(audio)
                audio = audio.unsqueeze(0)  # [B(1), T_time]

                # Load pre-computed mel from disk
                mel = np.load(
                    os.path.join(
                        self.base_mels_path,
                        os.path.splitext(os.path.split(filename)[-1])[0] + ".npy",
                    )
                )
                mel = torch.from_numpy(mel)

                if len(mel.shape) < 3:
                    mel = mel.unsqueeze(0)  # ensure [B, C, T]

                if self.split:
                    frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                    if audio.size(1) >= self.segment_size:
                        mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                        mel = mel[:, :, mel_start : mel_start + frames_per_seg]
                        audio = audio[
                            :,
                            mel_start
                            * self.hop_size : (mel_start + frames_per_seg)
                            * self.hop_size,
                        ]

                    # Pad pre-computed mel and audio to match length to ensuring fine-tuning without error.
                    # NOTE: this may introduce a single-frame misalignment of the <pre-computed mel, audio>
                    # To remove possible misalignment, it is recommended to prepare the <pre-computed mel, audio> pair where the audio length is the integer multiple of self.hop_size
                    mel = torch.nn.functional.pad(
                        mel, (0, frames_per_seg - mel.size(2)), "constant"
                    )
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

            # Compute mel_loss used by spectral regression objective. Uses self.fmax_loss instead (usually None)
            mel_loss = mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax_loss,
                center=False,
            )  # [B(1), self.num_mels, self.segment_size // self.hop_size]

            # Shape sanity checks
            assert (
                audio.shape[1] == mel.shape[2] * self.hop_size
                and audio.shape[1] == mel_loss.shape[2] * self.hop_size
            ), f"Audio length must be mel frame length * hop_size. Got audio shape {audio.shape} mel shape {mel.shape} mel_loss shape {mel_loss.shape}"

            return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

        # If it encounters error during loading the data, skip this sample and load random other sample to the batch
        except Exception as e:
            if self.fine_tuning:
                raise e  # Terminate training if it is fine-tuning. The dataset should have been prepared properly.
            else:
                print(
                    f"[WARNING] Failed to load waveform, skipping! filename: {filename} Error: {e}"
                )
                return self[random.randrange(len(self))]

    def __len__(self):
        return len(self.audio_files)
