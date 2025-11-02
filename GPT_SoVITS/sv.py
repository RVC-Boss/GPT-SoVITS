from functools import lru_cache

import torch
import torchaudio

from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2

sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"

torchaudio.compliance.kaldi.get_mel_banks = lru_cache(maxsize=128)(torchaudio.compliance.kaldi.get_mel_banks)


class SV:
    def __init__(self, device: torch.device, fp16: bool = False):
        pretrained_state = torch.load(sv_path, map_location="cpu")
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        self.dtype = torch.float16 if fp16 else torch.float32
        if fp16 is False:
            self.embedding_model = self.embedding_model.to(device)
        else:
            self.embedding_model = self.embedding_model.half().to(device)

    def compute_embedding(self, wav: torch.Tensor):
        if not torch.cuda.is_available():
            wav = wav.float()
        feat = torch.stack(
            [
                torchaudio.compliance.kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0)
                for wav0 in wav
            ]
        ).to(self.dtype)
        sv_emb: torch.Tensor = self.embedding_model.forward3(feat)
        return sv_emb
