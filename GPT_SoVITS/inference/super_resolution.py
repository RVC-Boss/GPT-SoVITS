"""AP-BWE adapter used for inference-time bandwidth extension."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio.functional as aF

from GPT_SoVITS.runtime.paths import AP_BWE_DIR, resolve_ap_bwe_checkpoint
from GPT_SoVITS.vendor.ap_bwe.audio_ops import amp_pha_istft, amp_pha_stft
from GPT_SoVITS.vendor.ap_bwe.models.model import APNet_BWE_Model


AP_BWE_main_dir_path = str(AP_BWE_DIR)


class AP_BWE:
    def __init__(self, device, DictToAttrRecursive, checkpoint_file=None):
        checkpoint_path = resolve_ap_bwe_checkpoint(checkpoint_file)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        config_path = Path(checkpoint_path).parent / "config.json"
        with config_path.open(encoding="utf-8") as file:
            json_config = json.load(file)
        h = DictToAttrRecursive(json_config)
        model = APNet_BWE_Model(h).to(device)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict["generator"])
        model.eval()
        self.device = device
        self.model = model
        self.h = h

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        self.device = self.model.conv_pre_mag.weight.device
        return self

    def __call__(self, audio, orig_sampling_rate):
        with torch.no_grad():
            audio = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=self.h.hr_sampling_rate)
            amp_nb, pha_nb, _ = amp_pha_stft(audio, self.h.n_fft, self.h.hop_size, self.h.win_size)
            amp_wb_g, pha_wb_g, _ = self.model(amp_nb, pha_nb)
            audio_hr_g = amp_pha_istft(
                amp_wb_g,
                pha_wb_g,
                self.h.n_fft,
                self.h.hop_size,
                self.h.win_size,
            )
            return audio_hr_g.squeeze().cpu().numpy(), self.h.hr_sampling_rate
