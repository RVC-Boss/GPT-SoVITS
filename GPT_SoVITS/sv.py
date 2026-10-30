import torch

from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2
from GPT_SoVITS.eres2net import kaldi as Kaldi

sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"


class SV:
    def __init__(self, device, is_half):
        pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=False)
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        # The original speaker checkpoint also contains the unused classification projection.
        feature_state = {key: value for key, value in pretrained_state.items() if key not in {"seg_1.weight", "seg_1.bias"}}
        embedding_model.load_state_dict(feature_state, strict=True)
        embedding_model.eval()
        self.embedding_model = embedding_model
        if not is_half:
            self.embedding_model = self.embedding_model.to(device)
        else:
            self.embedding_model = self.embedding_model.half().to(device)
        self.is_half = is_half

    def compute_embedding3(self, wav):
        with torch.no_grad():
            if self.is_half:
                wav = wav.half()
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            )
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb
