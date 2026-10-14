import os

import torch
from module.models import CFM


V5_TEMPERATURE = 0.875
V5_REFERENCE_ANCHOR = 1.0
V5_ROLLING_TAIL_FRAMES = 32
V5_MAX_TARGET_CHUNK_FRAMES = 640
V5_REFERENCE_FRAMES = 500
V5_TOTAL_CHUNK_FRAMES = 1000
V5_VERSIONS = {"v5dev", "v5turbo"}


class CFMV5(CFM):
    def __init__(self, in_channels, dit):
        super().__init__(in_channels, dit)
        self.use_step_embedding = False
        self.cfg_drop_text = False
        self.noise_temperature = V5_TEMPERATURE
        self.use_static_cache = os.environ.get(
            "GSVV5_CFM_STATIC_CACHE", os.environ.get("GSVV4_CFM_STATIC_CACHE", "1")
        ) != "0"

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, n_timesteps, temperature=V5_TEMPERATURE,
                  inference_cfg_rate=1.30):
        return super().inference(mu, x_lens, prompt, n_timesteps, temperature, inference_cfg_rate)

    def forward(self, x1, x_lens, prompt_lens, mu, use_grad_ckpt):
        b = x1.shape[0]
        t = torch.rand([b], device=mu.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1, device=mu.device)
        vt = x1 - x0
        xt = x0 + t[:, None, None] * vt
        prompt = torch.zeros_like(x1)
        for i in range(b):
            prompt[i, :, : prompt_lens[i]] = x1[i, :, : prompt_lens[i]]
            xt[i, :, : prompt_lens[i]] = 0
        vt_pred = self.estimator(
            xt, prompt, x_lens, t, text0=mu, use_grad_ckpt=use_grad_ckpt,
        ).transpose(2, 1)
        loss = 0
        for i in range(b):
            loss += self.criterion(vt_pred[i, :, prompt_lens[i] : x_lens[i]], vt[i, :, prompt_lens[i] : x_lens[i]])
        return loss / b


@torch.inference_mode()
def synthesize_v5_mel(model, reference_features, target_features, reference_mel,
                      sample_steps=None, cfg_rate=None):
    steps = (4 if model.version == "v5turbo" else 32) if sample_steps is None else int(sample_steps)
    cfg = (1.30 if model.version == "v5dev" else 0.0) if cfg_rate is None else float(cfg_rate)
    reference_frames = min(reference_mel.shape[-1], reference_features.shape[-1])
    reference_mel = reference_mel[..., :reference_frames][..., -V5_REFERENCE_FRAMES:]
    reference_features = reference_features[..., :reference_frames][..., -V5_REFERENCE_FRAMES:]
    reference_frames = reference_mel.shape[-1]
    chunk_frames = min(V5_TOTAL_CHUNK_FRAMES - reference_frames, V5_MAX_TARGET_CHUNK_FRAMES)
    original_mel = reference_mel
    original_features = reference_features
    rolling_mel = reference_mel
    rolling_features = reference_features
    results = []
    for index, start in enumerate(range(0, target_features.shape[-1], chunk_frames)):
        target = target_features[..., start:start + chunk_frames]
        if index == 0:
            prompt = original_mel
            features = original_features
        else:
            tail = min(V5_ROLLING_TAIL_FRAMES, reference_frames,
                       rolling_mel.shape[-1], rolling_features.shape[-1])
            prefix = reference_frames - tail
            prompt = torch.cat((original_mel[..., :prefix], rolling_mel[..., -tail:]), dim=-1)
            features = torch.cat((original_features[..., :prefix], rolling_features[..., -tail:]), dim=-1)
        mu = torch.cat((features, target), dim=-1).transpose(2, 1)
        lengths = torch.full((mu.shape[0],), mu.shape[1], dtype=torch.long, device=mu.device)
        generated = model.cfm.inference(
            mu, lengths, prompt, steps, inference_cfg_rate=cfg,
        )[..., prompt.shape[-1]:]
        results.append(generated)
        rolling_mel = generated[..., -reference_frames:].to(reference_mel.dtype)
        rolling_features = target[..., -reference_frames:]
    return torch.cat(results, dim=-1)
