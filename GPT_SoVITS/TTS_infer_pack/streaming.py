"""Opt-in API v2 semantic streaming. Ordinary inference remains in TTS.run."""
import math
import threading
import time
import torch
from tools.acceleration import create_acceleration


def validate_streaming_options(req, version, api_mode=False):
    if not req.get("gpt_streaming", False):
        return
    if version not in {"v2ProPlus", "v5turbo", "v5dev"}:
        raise ValueError("gpt_streaming supports v2ProPlus, v5turbo and v5dev")
    mode = req.get("streaming_mode", False)
    if (api_mode and (type(mode) is not int or mode not in (2, 3))) or (not api_mode and mode not in (True, 2, 3)):
        raise ValueError("gpt_streaming requires streaming_mode=2 or 3")
    if req.get("text_split_method", "cut5") != "cut0" or req.get("batch_size", 1) != 1:
        raise ValueError("gpt_streaming requires text_split_method=cut0 and batch_size=1")
    if req.get("if_freeze", False):
        raise ValueError("gpt_streaming cannot freeze semantic tokens")
    if req.get("streaming_window_mode", "fixed") not in {"fixed", "hybrid"}:
        raise ValueError("streaming_window_mode must be fixed or hybrid")
    if req.get("media_type", "wav") not in {"wav", "raw"}:
        raise ValueError("gpt_streaming supports wav or raw PCM")
    if version.startswith("v5") and not str(req.get("prompt_text") or "").strip():
        raise ValueError("V5 semantic streaming requires prompt_text")
    if version.startswith("v5") and req.get("aux_ref_audio_paths"):
        raise ValueError("V5 semantic streaming supports one reference audio")
    for key, default, lower, upper in (
        ("streaming_chunk_seconds", 2.0, 0.2, 5.0),
        ("streaming_lookahead_seconds", 0.12, 0.0, 0.5),
        ("streaming_overlap_seconds", 0.04, 0.0, 0.2),
        ("gpt_streaming_lookahead_seconds", 0.5, 0.16, 2.0),
        ("speed_factor", 1.0, 0.1, 4.0),
        ("fragment_interval", 0.3, 0.0, 10.0),
    ):
        value = float(req.get(key, default))
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key} must be between {lower} and {upper}")
    if req.get("min_chunk_length", 16) < 1 or req.get("overlap_length", 2) < 1:
        raise ValueError("Streaming token intervals and overlaps must be positive")
    steps = req.get("sample_steps")
    if steps is not None and steps < 1:
        raise ValueError("sample_steps must be positive")
    cfg = req.get("cfg_rate")
    if cfg is not None and (not math.isfinite(float(cfg)) or float(cfg) < 0):
        raise ValueError("cfg_rate must be finite and nonnegative")


class PipelineStop:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cancelled = threading.Event()

    def is_set(self):
        return self.cancelled.is_set() or self.pipeline.stop_flag

    def set(self):
        self.cancelled.set()


def semantic_updates(pipeline, x, prompt, bert, req, stop):
    # A separate decoder leaves the ordinary context and cache unchanged.
    acceleration = pipeline.streaming_accel
    if acceleration is None:
        acceleration = create_acceleration(
            pipeline.configs.t2s_weights_path, device=pipeline.configs.device,
            dtype=pipeline.precision, max_batch_size=1, max_seq_length=2048,
            use_cuda_graph=req.get("use_cuda_graph", True),
            use_flash_attention=req.get("use_flash_attention", True))
        pipeline.streaming_accel = acceleration
    if acceleration is None:
        raise ValueError("gpt_streaming requires a supported AR acceleration backend")
    stream_state = {"poll_partial": req.get("streaming_window_mode", "fixed") == "hybrid"}
    updates = acceleration.infer_stream(
        x, prompt, bert, chunk_length=min(16, req.get("min_chunk_length", 16)),
        max_decode_steps=min(1500, pipeline.configs.hz * pipeline.configs.max_sec),
        top_k=req.get("top_k", 15), top_p=req.get("top_p", 1),
        temperature=req.get("temperature", 1),
        early_stop_num=pipeline.configs.hz * pipeline.configs.max_sec,
        repetition_penalty=req.get("repetition_penalty", 1.35),
        use_cuda_graph=req.get("use_cuda_graph", True),
        use_flash_attention=req.get("use_flash_attention", True), stop=stop, stream_state=stream_state)
    try:
        yield from updates
    finally:
        stop.set()
        updates.close()


class StreamingWindowSchedule:
    """Grow only when emitted PCM leaves time for another block.

    The clock starts before waiting for AR. Observations include AR waits,
    acoustic decoding, CPU transfer and time spent yielding to the caller.
    Playback slack assumes immediate playback of the first emitted PCM; it
    does not claim to know a remote player's queue or network delays.
    """
    def __init__(self, initial_units, max_units):
        self.initial = initial_units
        self.maximum = max(initial_units, max_units)
        self.last_units = initial_units
        self.started = time.perf_counter()
        self.first_pcm_at = None
        self.audio_seconds = 0.0
        self.cost_per_unit = None

    def choose(self, available_units):
        if self.first_pcm_at is None:
            return min(self.initial, available_units)
        slack = self.audio_seconds - (time.perf_counter() - self.first_pcm_at)
        growth_limit = min(self.maximum, max(self.initial, round(self.last_units * 1.5)))
        candidates = {self.initial, self.maximum}
        size = self.initial
        while size < self.maximum:
            size = min(self.maximum, max(size + 1, round(size * 1.5)))
            candidates.add(size)
        for size in sorted(candidates, reverse=True):
            if size > min(available_units, growth_limit):
                continue
            if self.cost_per_unit * size * 1.25 + 0.1 <= slack:
                return size
        # A smaller known block is preferable to waiting for a larger prefix.
        return min(self.initial, available_units)

    def observe(self, units, samples, sample_rate):
        now = time.perf_counter()
        rate = (now - self.started) / max(1, units)
        self.cost_per_unit = rate if self.cost_per_unit is None else max(rate, self.cost_per_unit * 0.8)
        if self.first_pcm_at is None:
            self.first_pcm_at = now
        self.audio_seconds += samples / sample_rate
        self.last_units = units
        self.started = now


def v2_token_chunks(updates, chunk_length, mute_emb_sim_matrix=None, threshold=0.0, schedule=None):
    """Convert cumulative AR prefixes to non-overlapping V2 decoder deltas."""
    cursor = previous_length = 0
    try:
        for tokens, finished in updates:
            tokens = tokens.reshape(-1)
            if tokens.numel() < previous_length:
                raise RuntimeError("Semantic stream moved backwards")
            previous_length = tokens.numel()
            if finished and schedule is None:
                tail = tokens[cursor:]
                yield (tail.unsqueeze(0) if tail.numel() else None), True
                return
            while cursor < tokens.numel():
                available = tokens.numel() - cursor
                length = schedule.choose(available if finished else schedule.maximum) if schedule is not None else chunk_length
                if not finished and available < length + (2 if mute_emb_sim_matrix is not None else 0):
                    break
                if schedule is not None and not finished and available < chunk_length:
                    break
                if finished and available <= length:
                    yield tokens[cursor:].unsqueeze(0), True
                    return
                if mute_emb_sim_matrix is None:
                    end = cursor + length
                else:
                    similarity = mute_emb_sim_matrix[tokens[cursor:]]
                    pairs = (similarity[:-1] >= threshold) & (similarity[1:] >= threshold)
                    pairs[:chunk_length-1] = False
                    if schedule is not None:
                        pairs[length:] = False
                    if not pairs.any():
                        if schedule is None:
                            break
                        end = cursor + length
                    else:
                        scores = (similarity[:-1] + similarity[1:]).masked_fill(~pairs, -torch.inf)
                        offset = int(scores.argmax().item())
                        end = cursor + offset + 1
                final = finished and end == tokens.numel()
                yield tokens[cursor:end].unsqueeze(0), final
                cursor = end
                if final:
                    return
            if finished:
                yield None, True
                return
    finally:
        updates.close()


def v5_reference(pipeline):
    from .TTS import resample, mel_fn_v4, norm_spec
    cache = pipeline.prompt_cache
    spec = cache["refer_spec"][0][0].to(pipeline.configs.device, dtype=pipeline.precision)
    prompt = cache["prompt_semantic"].reshape(1, 1, -1).to(pipeline.configs.device)
    phones = torch.tensor(cache["phones"], device=pipeline.configs.device).unsqueeze(0)
    features, ge = pipeline.vits_model.decode_encp(prompt, phones, spec)
    audio = cache["raw_audio"].to(pipeline.configs.device).float().mean(0, keepdim=True)
    if cache["raw_sr"] != 32000:
        audio = resample(audio, cache["raw_sr"], 32000, pipeline.configs.device)
    mel = norm_spec(mel_fn_v4(audio)).to(pipeline.precision)
    limit = min(mel.shape[-1], features.shape[-1])
    count = min(500, limit)
    return (cache["phones"], cache["bert_features"], prompt,
            spec, features[..., :limit][..., -count:], ge, mel[..., :limit][..., -count:])


@torch.inference_mode()
def stream_v5_windows(vq_model, vocoder, denorm_spec, device, token_updates, phones, reference, req, stop):
    """Yield PCM once a core window plus future semantic context is available.

    The encoder has noncausal attention: finite lookahead trades full-utterance
    context for latency. Emitted samples are never regenerated or sent twice.
    """
    core = max(1, round(req["streaming_chunk_seconds"] * 100))
    hybrid = req.get("streaming_window_mode", "fixed") == "hybrid"
    future = round(req["streaming_lookahead_seconds"] * 100)
    holdback = max(16, round(req["gpt_streaming_lookahead_seconds"] * 100))
    fade_frames = min(round(req["streaming_overlap_seconds"] * 100), core - 1)
    fade = fade_frames * 480
    left_frames = max(24, fade_frames)
    ref_features, ref_mel = reference[4], reference[6]
    count = min(500, ref_features.shape[-1], ref_mel.shape[-1])
    ref_features, ref_mel = ref_features[..., -count:], ref_mel[..., -count:]
    if count + core + future > 1000:
        raise ValueError("Reference plus streaming window exceeds the V5 1000-frame context")
    # The AR sampler uses the global CUDA RNG. CFM must not advance that RNG
    # while AR is still decoding on its owner thread.
    rng = torch.Generator(device=device).manual_seed(torch.initial_seed())
    phone_tensor = torch.tensor(phones, device=device, dtype=torch.long).unsqueeze(0)
    alpha = torch.linspace(0, 1, fade, device=ref_mel.device, dtype=torch.float32) if fade else None
    history = ref_mel[..., :0]
    feature_history = ref_features[..., :0]
    pending = None
    cursor = previous_tokens = 0
    schedule = StreamingWindowSchedule(core, min(1000-count, 640)) if hybrid else None
    for tokens, finished in token_updates:
        if stop.is_set():
            return
        if tokens.numel() < previous_tokens:
            raise RuntimeError("Semantic stream moved backwards")
        previous_tokens = tokens.numel()
        if schedule is not None:
            core = schedule.choose(schedule.maximum)
        # V5: 25 Hz semantic tokens -> 100 Hz acoustic features at speed 1.
        window_future = 0 if hybrid and cursor else future
        needed = math.ceil((cursor + core + window_future + holdback) * req["speed_factor"] / 4)
        if not finished and tokens.numel() < needed:
            continue
        target, _ = vq_model.decode_encp(
            tokens.reshape(1, 1, -1).long(), phone_tensor,
            reference[3], reference[5], req["speed_factor"])
        total = target.shape[-1]
        safe_end = total if finished else max(0, total - window_future - holdback)
        while cursor < total and (finished or cursor + core <= safe_end):
            if stop.is_set():
                return
            if schedule is not None:
                core = schedule.choose(safe_end-cursor if finished else schedule.maximum)
                window_future = 0 if cursor else future
                if not finished and cursor+core > safe_end:
                    break
            end = min(cursor + core, total)
            if cursor == 0:
                prompt, features = ref_mel, ref_features
            else:
                tail = min(32, count, history.shape[-1])
                prefix = count - tail
                prompt = torch.cat((ref_mel[..., :prefix], history[..., -tail:]), -1)
                tail_features = feature_history[..., -tail:] if hybrid else target[..., cursor-tail:cursor]
                features = torch.cat((ref_features[..., :prefix], tail_features), -1)
            mu = torch.cat((features, target[..., cursor:min(end + window_future, total)]), -1).transpose(2, 1)
            lengths = torch.full((mu.shape[0],), mu.shape[1], device=mu.device, dtype=torch.long)
            generated = vq_model.cfm.inference(
                mu, lengths, prompt, req["sample_steps"],
                inference_cfg_rate=req["cfg_rate"], generator=rng
            )[..., prompt.shape[-1]:].to(ref_mel.dtype)
            accepted = generated[..., :end-cursor]
            left = history[..., -left_frames:]
            audio = vocoder(denorm_spec(torch.cat((left, generated), -1)))[0, 0].float()
            offset = left.shape[-1] * 480
            current = audio[offset:offset+(end-cursor)*480]
            if current.numel() != (end-cursor)*480:
                raise RuntimeError("Vocoder returned an incomplete acoustic window")
            prefix_audio = current[:0] if pending is None else (
                pending * (1-alpha) + audio[offset-fade:offset] * alpha)
            if fade and (not finished or end != total):
                # Keep at most one overlap; the next window supplies its right context.
                emitted = torch.cat((prefix_audio, current[:-fade]))
                pending = current[-fade:]
            else:
                emitted = torch.cat((prefix_audio, current))
                pending = None
            history = torch.cat((history, accepted), -1)[..., -max(left_frames, 32):]
            feature_history = torch.cat((feature_history, target[..., cursor:end]), -1)[..., -max(left_frames, 32):]
            units = end-cursor
            cursor = end
            if stop.is_set():
                return
            if not torch.isfinite(emitted).all():
                raise RuntimeError("Non-finite V5 audio")
            pcm = (emitted.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
            if schedule is not None:
                schedule.observe(units, pcm.size, 48000)
            yield pcm
        if finished:
            if cursor != total:
                raise RuntimeError("Semantic streaming did not flush all acoustic frames")
            # EOS may arrive exactly at a previously emitted core boundary.
            if pending is not None:
                yield (pending.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
            return
    if not stop.is_set():
        raise RuntimeError("Semantic stream ended without a final result")
