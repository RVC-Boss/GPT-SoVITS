import torch

from GPT_SoVITS.eres2net import kaldi
from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2

sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"


class SV:
    def __init__(self, device, is_half):
        pretrained_state = torch.load(sv_path, map_location="cpu")
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        if is_half is False:
            self.embedding_model = self.embedding_model.to(device)
        else:
            self.embedding_model = self.embedding_model.half().to(device)
        self.is_half = is_half

    def compute_embedding(self, wav):
        if self.is_half is True:
            wav = wav.half()
        feat = torch.stack(
            [kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
        )
        sv_emb = self.embedding_model.forward3(feat)
        return sv_emb
