from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

import torch
import torchaudio

from tools.AP_BWE.datasets1.dataset import amp_pha_istft, amp_pha_stft
from tools.AP_BWE.models.model import APNet_BWE_Model

AP_BWE_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AP_BWE")


class AP_BWE:
    def __init__(self, device, DictToAttrRecursive, checkpoint_file=None):
        if checkpoint_file is None:
            checkpoint_file = "{AP_BWE_dir_path}/24kto48k/g_24kto48k.zip"
            if os.path.exists(checkpoint_file) is False:
                raise FileNotFoundError()
        config_file = os.path.join(os.path.split(checkpoint_file)[0], "config.json")
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        # h = AttrDict(json_config)
        h = DictToAttrRecursive(json_config)
        model = APNet_BWE_Model(h).to(device)
        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict["generator"])
        model.eval()
        self.device = device
        self.model = model
        self.h = h

    def to(self, *arg, **kwargs):
        self.model.to(*arg, **kwargs)
        self.device = self.model.conv_pre_mag.weight.device
        return self

    def __call__(self, audio, orig_sampling_rate):
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sampling_rate, new_freq=self.h.hr_sampling_rate)
        amp_nb, pha_nb, com_nb = amp_pha_stft(audio, self.h.n_fft, self.h.hop_size, self.h.win_size)
        amp_wb_g, pha_wb_g, com_wb_g = self.model(amp_nb, pha_nb)
        audio_hr_g = amp_pha_istft(amp_wb_g, pha_wb_g, self.h.n_fft, self.h.hop_size, self.h.win_size)
        return audio_hr_g.squeeze().cpu().numpy(), self.h.hr_sampling_rate
