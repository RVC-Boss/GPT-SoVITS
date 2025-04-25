import os
import random
import torch
import torchaudio
import torch.utils.data
import torchaudio.functional as aF


def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):
    hann_window = torch.hann_window(win_size).to(audio.device)
    stft_spec = torch.stft(
        audio,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    log_amp = torch.log(torch.abs(stft_spec) + 1e-4)
    pha = torch.angle(stft_spec)

    com = torch.stack((torch.exp(log_amp) * torch.cos(pha), torch.exp(log_amp) * torch.sin(pha)), dim=-1)

    return log_amp, pha, com


def amp_pha_istft(log_amp, pha, n_fft, hop_size, win_size, center=True):
    amp = torch.exp(log_amp)
    com = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    audio = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return audio


def get_dataset_filelist(a):
    with open(a.input_training_file, "r", encoding="utf-8") as fi:
        training_indexes = [x.split("|")[0] for x in fi.read().split("\n") if len(x) > 0]

    with open(a.input_validation_file, "r", encoding="utf-8") as fi:
        validation_indexes = [x.split("|")[0] for x in fi.read().split("\n") if len(x) > 0]

    return training_indexes, validation_indexes


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_indexes,
        wavs_dir,
        segment_size,
        hr_sampling_rate,
        lr_sampling_rate,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
    ):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.wavs_dir = wavs_dir
        self.segment_size = segment_size
        self.hr_sampling_rate = hr_sampling_rate
        self.lr_sampling_rate = lr_sampling_rate
        self.split = split
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
            audio, orig_sampling_rate = torchaudio.load(os.path.join(self.wavs_dir, filename + ".wav"))
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        if orig_sampling_rate == self.hr_sampling_rate:
            audio_hr = audio
        else:
            audio_hr = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=self.hr_sampling_rate)

        audio_lr = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=self.lr_sampling_rate)
        audio_lr = aF.resample(audio_lr, orig_freq=self.lr_sampling_rate, new_freq=self.hr_sampling_rate)
        audio_lr = audio_lr[:, : audio_hr.size(1)]

        if self.split:
            if audio_hr.size(1) >= self.segment_size:
                max_audio_start = audio_hr.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio_hr = audio_hr[:, audio_start : audio_start + self.segment_size]
                audio_lr = audio_lr[:, audio_start : audio_start + self.segment_size]
            else:
                audio_hr = torch.nn.functional.pad(audio_hr, (0, self.segment_size - audio_hr.size(1)), "constant")
                audio_lr = torch.nn.functional.pad(audio_lr, (0, self.segment_size - audio_lr.size(1)), "constant")

        return (audio_hr.squeeze(), audio_lr.squeeze())

    def __len__(self):
        return len(self.audio_indexes)
