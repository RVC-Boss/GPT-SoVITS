import math
import os

import torch
from torch import nn


V5_TEMPERATURE = 0.875
V5_REFERENCE_ANCHOR = 1.0
V5_ROLLING_TAIL_FRAMES = 32
V5_MAX_TARGET_CHUNK_FRAMES = 640
V5_REFERENCE_FRAMES = 500
V5_TOTAL_CHUNK_FRAMES = 1000
V5_DEFAULT_CFG = 1.30
V5_REQUEST_DEFAULT_CFG = 0.0
V5_VERSIONS = {"v5", "v5dev", "v5turbo"}


def sampling_defaults(version, legacy_steps=32):
    if version == "v5turbo":
        return 4, 0.0
    if version in V5_VERSIONS:
        return 32, 1.30
    return legacy_steps, 0.0


def resolve_sampling(version, sample_steps=None, cfg_rate=None, legacy_steps=32):
    steps, cfg = sampling_defaults(version, legacy_steps)
    return (steps if sample_steps is None else int(sample_steps),
            cfg if cfg_rate is None else float(cfg_rate))


def validate_v5_sampling(sample_steps, cfg_rate):
    steps = int(sample_steps)
    cfg = float(cfg_rate)
    if steps not in (4, 8, 16, 32):
        raise ValueError("Euler steps must be one of 4, 8, 16, 32")
    if not math.isfinite(cfg) or cfg < 0 or cfg > 2:
        raise ValueError("CFG must be a finite value between 0 and 2")
    return steps, cfg


class CFMV5(nn.Module):
    def __init__(self, in_channels, dit):
        super().__init__()
        self.in_channels = in_channels
        self.estimator = dit
        self.use_conditioner_cache = True
        self.use_static_cache = os.environ.get(
            "GSVV5_CFM_STATIC_CACHE", os.environ.get("GSVV4_CFM_STATIC_CACHE", "1")
        ) != "0"

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, n_timesteps, temperature=V5_TEMPERATURE,
                  inference_cfg_rate=V5_DEFAULT_CFG):
        steps, cfg = validate_v5_sampling(n_timesteps, inference_cfg_rate)
        batch, frames = mu.shape[:2]
        prompt_len = prompt.shape[-1]
        # Match v4: noise, time and model inputs follow mu/model dtype.
        state_dtype = mu.dtype
        model_dtype = mu.dtype
        x = torch.randn(batch, self.in_channels, frames, device=mu.device, dtype=state_dtype) * V5_TEMPERATURE
        x[..., :prompt_len] = 0
        prompt_x = torch.zeros_like(x, dtype=model_dtype)
        prompt_x[..., :prompt_len] = prompt.to(model_dtype)
        condition = mu.transpose(2, 1)
        cache_enabled = bool(self.use_conditioner_cache and self.use_static_cache)
        cache = (self.estimator.prepare_static_cache(prompt_x, x_lens, condition)
                 if cache_enabled else None)
        text_cache = None
        step = 1.0 / steps
        for index in range(steps):
            time = torch.full((batch,), index * step, device=mu.device, dtype=state_dtype)
            velocity, text_embedding, _ = self.estimator(
                x, prompt_x, x_lens, time, None, condition,
                drop_audio_cond=False, drop_text=False, static_cache=cache,
                infer=True, text_cache=text_cache,
            )
            if self.use_conditioner_cache and cache is None:
                text_cache = text_embedding
            velocity = velocity.transpose(2, 1)
            if cfg > 1e-5:
                negative, _, _ = self.estimator(
                    x, prompt_x, x_lens, time, None, condition,
                    drop_audio_cond=True, drop_text=False, static_cache=cache,
                    infer=True, text_cache=text_cache,
                )
                negative = negative.transpose(2, 1)
                velocity = velocity + cfg * (velocity - negative)
            x = x + step * velocity
            x[..., :prompt_len] = 0
        return x


@torch.inference_mode()
def synthesize_v5_mel(model, reference_features, target_features, reference_mel,
                      sample_steps=None, cfg_rate=None):
    sample_steps, cfg_rate = resolve_sampling(model.version, sample_steps, cfg_rate)
    steps, cfg = validate_v5_sampling(sample_steps, cfg_rate)
    reference_frames = min(reference_mel.shape[-1], reference_features.shape[-1])
    if reference_frames < 1:
        raise ValueError("V5 requires nonempty reference mel and semantic features")
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
    if not results:
        raise ValueError("V5 requires nonempty target semantic features")
    return torch.cat(results, dim=-1)
