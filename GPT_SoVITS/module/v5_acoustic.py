"""Opt-in V5 acoustic scheduling; no padding, generated-output cache or new weights."""

from collections import OrderedDict, defaultdict

import torch


@torch.inference_mode()
def infer_equal_length(cfm, mu, prompt, noise, steps, cfg):
    lengths = torch.full((mu.shape[0],), mu.shape[1], dtype=torch.long, device=mu.device)
    return cfm.inference(mu, lengths, prompt, steps, inference_cfg_rate=cfg, initial_noise=noise, all_valid=True)


@torch.inference_mode()
def synthesize_v5_mel_batch(
    model,
    reference_features,
    targets,
    reference_mel,
    steps=4,
    cfg=0.0,
    max_batch=8,
    noises=None,
    stats=None,
    runner=None,
):
    """Batch independent windows, retaining the original within-segment dependency."""
    from module.models_v5 import (
        V5_MAX_TARGET_CHUNK_FRAMES,
        V5_REFERENCE_FRAMES,
        V5_ROLLING_TAIL_FRAMES,
        V5_TOTAL_CHUNK_FRAMES,
    )

    if max_batch < 1:
        raise ValueError("max_batch must be positive")
    if not targets or any(t.shape[-1] < 1 for t in targets):
        raise ValueError("nonempty targets required")
    frames = min(reference_features.shape[-1], reference_mel.shape[-1])
    if frames < 1:
        raise ValueError("nonempty reference required")
    ref_mel = reference_mel[..., :frames][..., -V5_REFERENCE_FRAMES:]
    ref_feat = reference_features[..., :frames][..., -V5_REFERENCE_FRAMES:]
    frames = ref_mel.shape[-1]
    chunk = min(V5_TOTAL_CHUNK_FRAMES - frames, V5_MAX_TARGET_CHUNK_FRAMES)
    # Allocate in the original segment/window order and original individual shapes.
    # Batching must not change the CUDA RNG draws, nor cache generated audio/noise.
    if noises is None:
        noises = [
            [
                torch.randn(1, model.cfm.in_channels, frames + part.shape[-1], device=target.device, dtype=target.dtype)
                for part in target.split(chunk, dim=-1)
            ]
            for target in targets
        ]
    chunks = [list(t.split(chunk, dim=-1)) for t in targets]
    results = [[] for _ in targets]
    rolling_mel = [ref_mel for _ in targets]
    rolling_feat = [ref_feat for _ in targets]
    runner = runner or infer_equal_length
    for wave in range(max(map(len, chunks))):
        buckets = defaultdict(list)
        for i, parts in enumerate(chunks):
            if wave < len(parts):
                buckets[parts[wave].shape[-1]].append(i)
        for length, indices in buckets.items():
            for begin in range(0, len(indices), max_batch):
                ids = indices[begin : begin + max_batch]
                prompts, conditions, ns = [], [], []
                for i in ids:
                    if wave == 0:
                        prompt, feat = ref_mel, ref_feat
                    else:
                        tail = min(V5_ROLLING_TAIL_FRAMES, frames, rolling_mel[i].shape[-1], rolling_feat[i].shape[-1])
                        prefix = frames - tail
                        prompt = torch.cat((ref_mel[..., :prefix], rolling_mel[i][..., -tail:]), -1)
                        feat = torch.cat((ref_feat[..., :prefix], rolling_feat[i][..., -tail:]), -1)
                    prompts.append(prompt)
                    conditions.append(torch.cat((feat, chunks[i][wave]), -1).transpose(2, 1))
                    ns.append(noises[i][wave])
                mu, prompt, noise = (torch.cat(xs, 0) for xs in (conditions, prompts, ns))
                generated = runner(model.cfm, mu, prompt, noise, steps, cfg)[..., frames:]
                if stats is not None:
                    stats.append({"batch": len(ids), "frames": frames + length, "wave": wave})
                for row, i in enumerate(ids):
                    # clone separates each output from graph/static/shared batch storage.
                    result = generated[row : row + 1].clone()
                    results[i].append(result)
                    rolling_mel[i] = result[..., -frames:].to(ref_mel.dtype)
                    rolling_feat[i] = chunks[i][wave][..., -frames:]
    return [torch.cat(xs, -1) for xs in results]


class V5AcousticGraphRunner:
    """Bounded per-pipeline graphs; callers must serialize use and copy outputs."""

    def __init__(self, max_entries=64):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0
        self.model_context = None
        self.pool = None

    @torch.inference_mode()
    def __call__(self, cfm, mu, prompt, noise, steps, cfg):
        if mu.device.type != "cuda":
            return infer_equal_length(cfm, mu, prompt, noise, steps, cfg)
        parameter = next(cfm.parameters())
        context = (id(cfm), parameter.data_ptr(), parameter.dtype, parameter.device)
        if self.model_context != context:
            self.entries.clear()
            self.pool = torch.cuda.graph_pool_handle()
            self.model_context = context
        # Graphs bind weight addresses: drop old entries after model/device/dtype changes.
        key = (
            id(cfm),
            tuple(mu.shape),
            tuple(prompt.shape),
            mu.dtype,
            mu.device,
            steps,
            cfg,
            cfm.use_conditioner_cache,
            cfm.use_static_cache,
            cfm.noise_temperature,
        )
        if key in self.entries:
            self.hits += 1
            entry = self.entries.pop(key)
            self.entries[key] = entry
        else:
            self.misses += 1
            static_mu, static_prompt, static_noise = mu.clone(), prompt.clone(), noise.clone()
            # Side-stream warmup initializes lazy kernels/rotary buffers before capture.
            stream = torch.cuda.Stream(device=mu.device)
            stream.wait_stream(torch.cuda.current_stream(mu.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    infer_equal_length(cfm, static_mu, static_prompt, static_noise, steps, cfg)
            torch.cuda.current_stream(mu.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                output = infer_equal_length(cfm, static_mu, static_prompt, static_noise, steps, cfg)
            entry = (static_mu, static_prompt, static_noise, graph, output)
            if len(self.entries) >= self.max_entries:
                self.entries.popitem(last=False)
                self.evictions += 1
            self.entries[key] = entry
        entry[0].copy_(mu)
        entry[1].copy_(prompt)
        entry[2].copy_(noise)
        entry[3].replay()
        # Caller immediately clones slices before the next shared-pool graph replay.
        return entry[4]

    def clear(self):
        self.entries.clear()
        self.pool = None
        self.model_context = None

    def stats(self):
        return {"entries": len(self.entries), "hits": self.hits, "misses": self.misses, "evictions": self.evictions}
