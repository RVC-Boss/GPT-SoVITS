from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from AR.models.utils import make_pad_mask_left, sample


def _sync_device(device: Any) -> None:
    try:
        device_str = str(device)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    except Exception:
        pass


@dataclass
class SchedulerRequestSpec:
    request_id: str
    ref_audio_path: Path
    prompt_text: str
    prompt_lang: str
    text: str
    text_lang: str
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    early_stop_num: int
    ready_step: int = 0


@dataclass
class T2SRequestState:
    request_id: str
    ref_audio_path: Path
    prompt_text: str
    prompt_lang: str
    text: str
    text_lang: str
    norm_prompt_text: str
    norm_text: str
    phones: torch.LongTensor
    prompt_phones: torch.LongTensor
    all_phones: torch.LongTensor
    all_bert_features: torch.Tensor
    prompt_semantic: torch.LongTensor
    refer_spec: Tuple[torch.Tensor, Optional[torch.Tensor]]
    raw_audio: torch.Tensor
    raw_sr: int
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    early_stop_num: int
    ready_step: int
    prepare_profile: Dict[str, float]


@dataclass
class T2SRunningRequest:
    state: T2SRequestState
    y_sequence: torch.LongTensor
    prefix_len: int
    decode_attn_mask: torch.Tensor
    k_cache: List[torch.Tensor]
    v_cache: List[torch.Tensor]
    step_idx: int


@dataclass
class T2SFinishedItem:
    request_id: str
    semantic_tokens: torch.LongTensor
    finish_idx: int
    finish_reason: str


@dataclass
class T2SActiveBatch:
    request_ids: List[str]
    states: List[T2SRequestState]
    x: torch.Tensor
    x_lens: torch.LongTensor
    y_sequences: List[torch.LongTensor]
    prefix_lens: torch.LongTensor
    xy_pos: torch.Tensor
    prefill_attn_mask: torch.Tensor
    decode_attn_mask: Optional[torch.Tensor]
    k_cache: Optional[List[torch.Tensor]]
    v_cache: Optional[List[torch.Tensor]]
    step_idx: int
    prefill_done: bool


def normalize_sentence(text: str, language: str) -> str:
    text = text.strip("\n").strip()
    if not text:
        return text
    if text[-1] not in {",", ".", "?", "!", "，", "。", "？", "！", "…", "；", ";", ":"}:
        text += "。" if language != "en" else "."
    return text


def prepare_request_state(
    tts: Any,
    spec: SchedulerRequestSpec,
) -> T2SRequestState:
    device = tts.configs.device
    prepare_start = time.perf_counter()
    _sync_device(device)
    prepare_sync_start = time.perf_counter()
    prompt_text = normalize_sentence(spec.prompt_text, spec.prompt_lang)
    text = spec.text.strip("\n")

    _sync_device(device)
    prompt_text_features_start = time.perf_counter()
    prompt_phones, prompt_bert_features, prompt_norm_text = tts.extract_text_features(prompt_text, spec.prompt_lang)
    _sync_device(device)
    prompt_text_features_ms = (time.perf_counter() - prompt_text_features_start) * 1000.0

    _sync_device(device)
    text_features_start = time.perf_counter()
    phones, bert_features, norm_text = tts.extract_text_features(text, spec.text_lang)
    _sync_device(device)
    text_features_ms = (time.perf_counter() - text_features_start) * 1000.0
    if phones is None:
        raise ValueError(f"{spec.request_id} text preprocessing returned no phones")

    _sync_device(device)
    prompt_semantic_start = time.perf_counter()
    prompt_semantic = tts.extract_prompt_semantic(str(spec.ref_audio_path)).long()
    _sync_device(device)
    prompt_semantic_ms = (time.perf_counter() - prompt_semantic_start) * 1000.0

    _sync_device(device)
    ref_spec_start = time.perf_counter()
    spec_audio, audio_16k, raw_audio, raw_sr = tts.extract_ref_spec(str(spec.ref_audio_path))
    _sync_device(device)
    ref_spec_ms = (time.perf_counter() - ref_spec_start) * 1000.0

    _sync_device(device)
    tensorize_start = time.perf_counter()
    phones_tensor = torch.LongTensor(phones).to(tts.configs.device)
    prompt_phones_tensor = torch.LongTensor(prompt_phones).to(tts.configs.device)
    all_phones = torch.LongTensor(prompt_phones + phones).to(tts.configs.device)
    all_bert_features = torch.cat([prompt_bert_features, bert_features], dim=1).to(
        dtype=tts.precision, device=tts.configs.device
    )
    _sync_device(device)
    tensorize_ms = (time.perf_counter() - tensorize_start) * 1000.0

    _sync_device(device)
    prepare_profile = {
        "prompt_text_features_ms": prompt_text_features_ms,
        "text_features_ms": text_features_ms,
        "prompt_semantic_ms": prompt_semantic_ms,
        "ref_spec_ms": ref_spec_ms,
        "tensorize_ms": tensorize_ms,
        "total_ms": (time.perf_counter() - prepare_sync_start) * 1000.0,
        "wall_total_ms": (time.perf_counter() - prepare_start) * 1000.0,
    }
    return T2SRequestState(
        request_id=spec.request_id,
        ref_audio_path=spec.ref_audio_path,
        prompt_text=prompt_text,
        prompt_lang=spec.prompt_lang,
        text=text,
        text_lang=spec.text_lang,
        norm_prompt_text=prompt_norm_text,
        norm_text=norm_text,
        phones=phones_tensor,
        prompt_phones=prompt_phones_tensor,
        all_phones=all_phones,
        all_bert_features=all_bert_features,
        prompt_semantic=prompt_semantic,
        refer_spec=(spec_audio, audio_16k),
        raw_audio=raw_audio,
        raw_sr=int(raw_sr),
        top_k=spec.top_k,
        top_p=spec.top_p,
        temperature=spec.temperature,
        repetition_penalty=spec.repetition_penalty,
        early_stop_num=spec.early_stop_num,
        ready_step=spec.ready_step,
        prepare_profile=prepare_profile,
    )


def _left_pad_hidden(hidden: torch.Tensor, target_len: int) -> torch.Tensor:
    if hidden.shape[0] >= target_len:
        return hidden
    return F.pad(hidden, (0, 0, target_len - hidden.shape[0], 0), value=0)


def _ensure_audio_pe(model: Any, max_position: int, dtype: torch.dtype, device: torch.device) -> None:
    required_len = max_position + 1
    if model.ar_audio_position.pe is not None and model.ar_audio_position.pe.size(1) >= required_len:
        if model.ar_audio_position.pe.dtype != dtype or model.ar_audio_position.pe.device != device:
            model.ar_audio_position.pe = model.ar_audio_position.pe.to(dtype=dtype, device=device)
        return
    model.ar_audio_position.extend_pe(
        torch.zeros(1, required_len, model.ar_audio_position.embedding_dim, device=device, dtype=dtype)
    )


def build_prefill_batch(model: Any, states: Sequence[T2SRequestState]) -> T2SActiveBatch:
    x_items: List[torch.Tensor] = []
    y_pos_items: List[torch.Tensor] = []
    x_lens: List[int] = []
    prefix_lens: List[int] = []
    y_sequences: List[torch.LongTensor] = []

    for state in states:
        text_emb = model.ar_text_embedding(state.all_phones.unsqueeze(0))
        bert_proj = model.bert_proj(state.all_bert_features.transpose(0, 1).unsqueeze(0))
        x_pos = model.ar_text_position(text_emb + bert_proj).squeeze(0)
        y_emb = model.ar_audio_embedding(state.prompt_semantic.unsqueeze(0))
        y_pos = model.ar_audio_position(y_emb).squeeze(0)
        x_items.append(x_pos)
        y_pos_items.append(y_pos)
        x_lens.append(x_pos.shape[0])
        prefix_lens.append(y_pos.shape[0])
        y_sequences.append(state.prompt_semantic.clone())

    max_x_len = max(x_lens)
    max_prefix_len = max(prefix_lens)
    x_batch = torch.stack([_left_pad_hidden(item, max_x_len) for item in x_items], dim=0)
    y_pos_batch = torch.stack([_left_pad_hidden(item, max_prefix_len) for item in y_pos_items], dim=0)
    xy_pos = torch.cat([x_batch, y_pos_batch], dim=1)

    device = x_batch.device
    x_lens_tensor = torch.LongTensor(x_lens).to(device)
    prefix_lens_tensor = torch.LongTensor(prefix_lens).to(device)
    batch_size = len(states)
    src_len = max_x_len + max_prefix_len

    x_padding_mask = make_pad_mask_left(x_lens_tensor, max_x_len)
    y_padding_mask = make_pad_mask_left(prefix_lens_tensor, max_prefix_len)
    padding_mask = torch.cat([x_padding_mask, y_padding_mask], dim=1)
    x_mask = F.pad(torch.zeros(max_x_len, max_x_len, dtype=torch.bool, device=device), (0, max_prefix_len), value=True)
    y_mask = F.pad(
        torch.triu(torch.ones(max_prefix_len, max_prefix_len, dtype=torch.bool, device=device), diagonal=1),
        (max_x_len, 0),
        value=False,
    )
    causal_mask = torch.cat([x_mask, y_mask], dim=0).view(1, src_len, src_len).repeat(batch_size, 1, 1)
    padding_mask = padding_mask.view(batch_size, 1, src_len).repeat(1, src_len, 1)
    attn_mask = causal_mask.logical_or(padding_mask)
    attn_mask = attn_mask.unsqueeze(1).expand(-1, model.num_head, -1, -1).bool()

    return T2SActiveBatch(
        request_ids=[state.request_id for state in states],
        states=list(states),
        x=x_batch,
        x_lens=x_lens_tensor,
        y_sequences=y_sequences,
        prefix_lens=prefix_lens_tensor,
        xy_pos=xy_pos,
        prefill_attn_mask=attn_mask,
        decode_attn_mask=None,
        k_cache=None,
        v_cache=None,
        step_idx=0,
        prefill_done=False,
    )


def build_next_xy_pos(model: Any, y_sequences: Sequence[torch.LongTensor]) -> torch.Tensor:
    last_tokens = torch.stack([seq[-1:] for seq in y_sequences], dim=0)
    y_emb = model.ar_audio_embedding(last_tokens)
    position_ids = torch.LongTensor([int(seq.shape[0] - 1) for seq in y_sequences]).to(y_emb.device)
    _ensure_audio_pe(model, int(position_ids.max().item()), y_emb.dtype, y_emb.device)
    pos_emb = model.ar_audio_position.pe[0].index_select(0, position_ids).unsqueeze(1)
    return y_emb * model.ar_audio_position.x_scale + model.ar_audio_position.alpha * pos_emb.to(
        dtype=y_emb.dtype, device=y_emb.device
    )


def _sample_per_request(
    model: Any,
    active_batch: T2SActiveBatch,
    logits: torch.Tensor,
    max_steps: int,
) -> Tuple[List[T2SFinishedItem], List[int], List[torch.LongTensor]]:
    finished_items: List[T2SFinishedItem] = []
    keep_indices: List[int] = []
    updated_sequences: List[torch.LongTensor] = []

    step_idx = active_batch.step_idx
    for batch_index, state in enumerate(active_batch.states):
        logits_i = logits[batch_index : batch_index + 1].clone()
        current_history = active_batch.y_sequences[batch_index]
        sampled = sample(
            logits_i,
            current_history.unsqueeze(0),
            top_k=state.top_k,
            top_p=state.top_p,
            repetition_penalty=state.repetition_penalty,
            temperature=state.temperature,
        )[0]
        sampled_token = int(sampled[0, 0].item())
        argmax_token = int(torch.argmax(logits[batch_index], dim=-1).item())
        new_history = torch.cat([current_history, sampled.view(-1)], dim=0)

        finish_reason: Optional[str] = None
        if state.early_stop_num != -1 and (new_history.shape[0] - int(active_batch.prefix_lens[batch_index].item())) > state.early_stop_num:
            finish_reason = "early_stop"
        elif step_idx + 1 >= max_steps:
            finish_reason = "max_step"
        elif sampled_token == model.EOS:
            finish_reason = "eos_sample"
        elif argmax_token == model.EOS:
            finish_reason = "eos_argmax"

        if finish_reason is not None:
            finished_items.append(
                T2SFinishedItem(
                    request_id=state.request_id,
                    semantic_tokens=new_history[:-1].clone(),
                    finish_idx=step_idx,
                    finish_reason=finish_reason,
                )
            )
        else:
            keep_indices.append(batch_index)
            updated_sequences.append(new_history)

    return finished_items, keep_indices, updated_sequences


def decode_one_step(
    model: Any,
    active_batch: T2SActiveBatch,
    max_steps: int,
) -> Tuple[Optional[T2SActiveBatch], List[T2SFinishedItem]]:
    if not active_batch.prefill_done:
        xy_dec, active_batch.k_cache, active_batch.v_cache = model.t2s_transformer.process_prompt(
            active_batch.xy_pos, active_batch.prefill_attn_mask, None
        )
        active_batch.decode_attn_mask = F.pad(
            active_batch.prefill_attn_mask[:, :, -1].unsqueeze(-2),
            (0, 1),
            value=False,
        )
        active_batch.prefill_done = True
    else:
        xy_dec, active_batch.k_cache, active_batch.v_cache = model.t2s_transformer.decode_next_token(
            active_batch.xy_pos,
            active_batch.k_cache,
            active_batch.v_cache,
            active_batch.decode_attn_mask,
        )
        if active_batch.decode_attn_mask is not None:
            active_batch.decode_attn_mask = F.pad(active_batch.decode_attn_mask, (0, 1), value=False)

    logits = model.ar_predict_layer(xy_dec[:, -1])
    if active_batch.step_idx < 11:
        logits = logits[:, :-1]

    finished_items, keep_indices, updated_sequences = _sample_per_request(model, active_batch, logits, max_steps=max_steps)
    if len(keep_indices) == 0:
        return None, finished_items

    device = logits.device
    keep_tensor = torch.LongTensor(keep_indices).to(device)
    active_batch.request_ids = [active_batch.request_ids[i] for i in keep_indices]
    active_batch.states = [active_batch.states[i] for i in keep_indices]
    active_batch.y_sequences = updated_sequences
    active_batch.prefix_lens = torch.index_select(active_batch.prefix_lens, dim=0, index=keep_tensor)

    if active_batch.decode_attn_mask is not None:
        active_batch.decode_attn_mask = torch.index_select(active_batch.decode_attn_mask, dim=0, index=keep_tensor)
    if active_batch.k_cache is not None and active_batch.v_cache is not None:
        for cache_index in range(len(active_batch.k_cache)):
            active_batch.k_cache[cache_index] = torch.index_select(active_batch.k_cache[cache_index], dim=0, index=keep_tensor)
            active_batch.v_cache[cache_index] = torch.index_select(active_batch.v_cache[cache_index], dim=0, index=keep_tensor)

    active_batch.xy_pos = build_next_xy_pos(model, active_batch.y_sequences)
    active_batch.step_idx += 1
    return active_batch, finished_items


def run_scheduler_batch(
    model: Any,
    states: Sequence[T2SRequestState],
    max_steps: int,
) -> List[T2SFinishedItem]:
    return run_scheduler_continuous(model, states, max_steps=max_steps)


def _pad_cache_left(cache: torch.Tensor, target_len: int) -> torch.Tensor:
    pad_len = target_len - cache.shape[1]
    if pad_len <= 0:
        return cache
    return F.pad(cache, (0, 0, pad_len, 0), value=0)


def _pad_decode_mask_left(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    pad_len = target_len - mask.shape[-1]
    if pad_len <= 0:
        return mask
    return F.pad(mask, (pad_len, 0), value=True)


def run_prefill_step(
    model: Any,
    states: Sequence[T2SRequestState],
    max_steps: int,
) -> Tuple[List[T2SRunningRequest], List[T2SFinishedItem]]:
    if not states:
        return [], []

    active_batch = build_prefill_batch(model, states)
    xy_dec, k_cache, v_cache = model.t2s_transformer.process_prompt(active_batch.xy_pos, active_batch.prefill_attn_mask, None)
    decode_attn_mask = F.pad(
        active_batch.prefill_attn_mask[:, :, -1].unsqueeze(-2),
        (0, 1),
        value=False,
    )
    logits = model.ar_predict_layer(xy_dec[:, -1])

    running_requests: List[T2SRunningRequest] = []
    finished_items: List[T2SFinishedItem] = []

    for batch_index, state in enumerate(states):
        logits_i = logits[batch_index : batch_index + 1].clone()
        if 0 < 11:
            logits_i = logits_i[:, :-1]
        current_history = active_batch.y_sequences[batch_index]
        sampled = sample(
            logits_i,
            current_history.unsqueeze(0),
            top_k=state.top_k,
            top_p=state.top_p,
            repetition_penalty=state.repetition_penalty,
            temperature=state.temperature,
        )[0]
        sampled_token = int(sampled[0, 0].item())
        argmax_token = int(torch.argmax(logits_i[0], dim=-1).item())
        new_history = torch.cat([current_history, sampled.view(-1)], dim=0)
        prefix_len = int(active_batch.prefix_lens[batch_index].item())

        finish_reason: Optional[str] = None
        if state.early_stop_num != -1 and (new_history.shape[0] - prefix_len) > state.early_stop_num:
            finish_reason = "early_stop"
        elif 1 >= max_steps:
            finish_reason = "max_step"
        elif sampled_token == model.EOS:
            finish_reason = "eos_sample"
        elif argmax_token == model.EOS:
            finish_reason = "eos_argmax"

        if finish_reason is not None:
            finished_items.append(
                T2SFinishedItem(
                    request_id=state.request_id,
                    semantic_tokens=new_history[:-1].clone(),
                    finish_idx=0,
                    finish_reason=finish_reason,
                )
            )
            continue

        real_kv_len = int(active_batch.x_lens[batch_index].item()) + prefix_len
        request_k_cache = [layer[batch_index : batch_index + 1, -real_kv_len:, :].clone() for layer in k_cache]
        request_v_cache = [layer[batch_index : batch_index + 1, -real_kv_len:, :].clone() for layer in v_cache]

        running_requests.append(
            T2SRunningRequest(
                state=state,
                y_sequence=new_history,
                prefix_len=prefix_len,
                decode_attn_mask=decode_attn_mask[batch_index : batch_index + 1].clone(),
                k_cache=request_k_cache,
                v_cache=request_v_cache,
                step_idx=1,
            )
        )

    return running_requests, finished_items


def _build_decode_batch_from_running(
    model: Any,
    running_requests: Sequence[T2SRunningRequest],
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    xy_pos = build_next_xy_pos(model, [item.y_sequence for item in running_requests])
    max_kv_len = max(item.k_cache[0].shape[1] for item in running_requests)
    max_mask_len = max(item.decode_attn_mask.shape[-1] for item in running_requests)
    num_layers = len(running_requests[0].k_cache)

    batched_k_cache: List[torch.Tensor] = []
    batched_v_cache: List[torch.Tensor] = []
    for layer_index in range(num_layers):
        batched_k_cache.append(
            torch.cat([_pad_cache_left(item.k_cache[layer_index], max_kv_len) for item in running_requests], dim=0)
        )
        batched_v_cache.append(
            torch.cat([_pad_cache_left(item.v_cache[layer_index], max_kv_len) for item in running_requests], dim=0)
        )

    batched_decode_attn_mask = torch.cat(
        [_pad_decode_mask_left(item.decode_attn_mask, max_mask_len) for item in running_requests],
        dim=0,
    )
    return xy_pos, batched_k_cache, batched_v_cache, batched_decode_attn_mask


def run_decode_step_for_running(
    model: Any,
    running_requests: Sequence[T2SRunningRequest],
    max_steps: int,
) -> Tuple[List[T2SRunningRequest], List[T2SFinishedItem]]:
    if not running_requests:
        return [], []

    xy_pos, batched_k_cache, batched_v_cache, batched_decode_attn_mask = _build_decode_batch_from_running(
        model, running_requests
    )
    xy_dec, next_k_cache, next_v_cache = model.t2s_transformer.decode_next_token(
        xy_pos,
        batched_k_cache,
        batched_v_cache,
        batched_decode_attn_mask,
    )
    logits = model.ar_predict_layer(xy_dec[:, -1])

    next_running: List[T2SRunningRequest] = []
    finished_items: List[T2SFinishedItem] = []

    for batch_index, running_request in enumerate(running_requests):
        current_idx = running_request.step_idx
        logits_i = logits[batch_index : batch_index + 1].clone()
        if current_idx < 11:
            logits_i = logits_i[:, :-1]
        sampled = sample(
            logits_i,
            running_request.y_sequence.unsqueeze(0),
            top_k=running_request.state.top_k,
            top_p=running_request.state.top_p,
            repetition_penalty=running_request.state.repetition_penalty,
            temperature=running_request.state.temperature,
        )[0]
        sampled_token = int(sampled[0, 0].item())
        argmax_token = int(torch.argmax(logits_i[0], dim=-1).item())
        new_history = torch.cat([running_request.y_sequence, sampled.view(-1)], dim=0)

        finish_reason: Optional[str] = None
        if running_request.state.early_stop_num != -1 and (new_history.shape[0] - running_request.prefix_len) > running_request.state.early_stop_num:
            finish_reason = "early_stop"
        elif current_idx + 1 >= max_steps:
            finish_reason = "max_step"
        elif sampled_token == model.EOS:
            finish_reason = "eos_sample"
        elif argmax_token == model.EOS:
            finish_reason = "eos_argmax"

        if finish_reason is not None:
            finished_items.append(
                T2SFinishedItem(
                    request_id=running_request.state.request_id,
                    semantic_tokens=new_history[:-1].clone(),
                    finish_idx=current_idx,
                    finish_reason=finish_reason,
                )
            )
            continue

        real_next_kv_len = running_request.k_cache[0].shape[1] + 1
        request_k_cache = [layer[batch_index : batch_index + 1, -real_next_kv_len:, :].clone() for layer in next_k_cache]
        request_v_cache = [layer[batch_index : batch_index + 1, -real_next_kv_len:, :].clone() for layer in next_v_cache]
        next_running.append(
            T2SRunningRequest(
                state=running_request.state,
                y_sequence=new_history,
                prefix_len=running_request.prefix_len,
                decode_attn_mask=F.pad(running_request.decode_attn_mask, (0, 1), value=False),
                k_cache=request_k_cache,
                v_cache=request_v_cache,
                step_idx=current_idx + 1,
            )
        )

    return next_running, finished_items


def run_scheduler_continuous(
    model: Any,
    states: Sequence[T2SRequestState],
    max_steps: int,
) -> List[T2SFinishedItem]:
    pending = sorted(states, key=lambda item: (item.ready_step, item.request_id))
    running_requests: List[T2SRunningRequest] = []
    finished: List[T2SFinishedItem] = []
    current_tick = 0

    while pending or running_requests:
        admitted: List[T2SRequestState] = []
        while pending and pending[0].ready_step <= current_tick:
            admitted.append(pending.pop(0))

        admitted_running, admitted_finished = run_prefill_step(model, admitted, max_steps=max_steps)
        finished.extend(admitted_finished)

        if running_requests:
            running_requests, step_finished = run_decode_step_for_running(
                model,
                running_requests,
                max_steps=max_steps,
            )
            finished.extend(step_finished)

        running_requests.extend(admitted_running)

        if not running_requests and pending:
            current_tick = max(current_tick + 1, pending[0].ready_step)
            continue

        current_tick += 1

    finished.sort(key=lambda item: item.request_id)
    return finished
