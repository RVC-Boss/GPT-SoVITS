"""Audio loading, resampling, and spectrogram helpers for inference."""

import numpy as np
import torch
import torchaudio

from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch


resample_transform_dict = {}
resampy_filter_cache = {}
spec_min = -12
spec_max = 2

def _resampy_gpu_filter(sr0, sr1, device):
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resampy_filter_cache:
        import resampy
        interp_win, precision, rolloff = resampy.filters.get_filter("kaiser_best")
        interp_win = torch.from_numpy(interp_win.astype(np.float32)).to(device)
        interp_delta = torch.diff(interp_win, append=interp_win[-1:])
        resampy_filter_cache[key] = (interp_win, interp_delta, int(precision))
    return resampy_filter_cache[key]

def resample(audio_tensor, sr0, sr1, device, match_librosa=False):
    if match_librosa:
        if int(sr0) == int(sr1):
            return audio_tensor.clone()
        ratio = float(sr1) / float(sr0)
        scale = min(1.0, ratio)
        n_orig = audio_tensor.shape[-1]
        n_out = int(n_orig * ratio)
        interp_win, interp_delta, num_table = _resampy_gpu_filter(sr0, sr1, audio_tensor.device)
        if ratio < 1.0:
            interp_win = interp_win * ratio
            interp_delta = interp_delta * ratio
        index_step = int(scale * num_table)
        nwin = interp_win.numel()
        t_all = torch.arange(n_out, device=audio_tensor.device, dtype=torch.float32) / ratio
        out = torch.zeros((audio_tensor.shape[0], n_out), device=audio_tensor.device, dtype=audio_tensor.dtype)
        chunk = 4096
        max_taps = (nwin + index_step - 1) // index_step
        for begin in range(0, n_out, chunk):
            t = t_all[begin:begin + chunk]
            n = torch.floor(t).long()
            frac = scale * (t - n.to(t.dtype))
            index_frac = frac * num_table
            offset = torch.floor(index_frac).long()
            eta = index_frac - offset.to(index_frac.dtype)
            ii = torch.arange(max_taps, device=audio_tensor.device).view(1, -1)
            li = offset.view(-1, 1) + ii * index_step
            lmax = torch.minimum(n + 1, (nwin - offset) // index_step).view(-1, 1)
            lm = ii < lmax
            li_safe = li.clamp(0, nwin - 1)
            lw = interp_win[li_safe] + eta.view(-1, 1) * interp_delta[li_safe]
            left_idx = (n.view(-1, 1) - ii).clamp(0, n_orig - 1)
            left = (lw * lm).to(audio_tensor.dtype)
            lv = audio_tensor[:, left_idx]
            frac_r = scale - frac
            index_frac = frac_r * num_table
            offset = torch.floor(index_frac).long()
            eta = index_frac - offset.to(index_frac.dtype)
            ri = offset.view(-1, 1) + ii * index_step
            rmax = torch.minimum(n_orig - n - 1, (nwin - offset) // index_step).view(-1, 1)
            rm = ii < rmax
            ri_safe = ri.clamp(0, nwin - 1)
            rw = interp_win[ri_safe] + eta.view(-1, 1) * interp_delta[ri_safe]
            right_idx = (n.view(-1, 1) + ii + 1).clamp(0, n_orig - 1)
            right = (rw * rm).to(audio_tensor.dtype)
            out[:, begin:begin + t.numel()] = (
                (lv * left.unsqueeze(0)).sum(-1)
                + (audio_tensor[:, right_idx] * right.unsqueeze(0)).sum(-1)
            )
        return out
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

def load_audio_gpu(path, target_sr, device, mono=True, dtype=torch.float32, match_librosa=False):
    """Load audio once, then resample and keep the signal on the target device."""
    audio, source_sr = torchaudio.load(path)
    audio = audio.to(device=device, dtype=dtype)
    if mono and audio.dim() == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if source_sr != target_sr:
        audio = resample(audio, source_sr, target_sr, device, match_librosa=match_librosa)
    return audio, target_sr

def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1

def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)


mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)
