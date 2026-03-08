#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import contextlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
gpt_sovits_dir = ROOT_DIR / "GPT_SoVITS"
if str(gpt_sovits_dir) not in sys.path:
    sys.path.append(str(gpt_sovits_dir))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import (  # noqa: E402
    SchedulerRequestSpec,
    T2SRequestState,
    T2SRunningRequest,
    _build_decode_batch_from_running,
    build_prefill_batch,
    prepare_request_state,
    run_decode_step_for_running,
    run_prefill_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break down T2S CUDA memory by stage and tensor groups.")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "GPT_SoVITS/configs/tts_infer.yaml")
    parser.add_argument("--request-manifest", type=Path, default=None)
    parser.add_argument("--scenario", type=str, default="auto4", choices=["auto4", "single"])
    parser.add_argument("--auto-count", type=int, default=4)
    parser.add_argument("--auto-wav-dir", type=Path, default=ROOT_DIR / "testwav")
    parser.add_argument("--auto-text-file", type=Path, default=ROOT_DIR / "test_cn.txt")
    parser.add_argument("--ref-audio", type=Path, default=ROOT_DIR / "test.wav")
    parser.add_argument("--prompt-text", type=str, default="是啊，主要是因为有调研需求的学者少了。")
    parser.add_argument("--prompt-lang", type=str, default="zh")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-file", type=Path, default=ROOT_DIR / "test_en.txt")
    parser.add_argument("--text-lang", type=str, default="zh")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--early-stop-num", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument("--worker-rounds", type=int, default=1)
    parser.add_argument("--worker-grad-mode", type=str, default="default", choices=["default", "inference_mode"])
    parser.add_argument("--compare-worker-grad-modes", action="store_true", default=False)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "TEMP/t2s_memory_breakdown/run1",
    )
    return parser.parse_args()


def set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _sync_device(device: Any) -> None:
    try:
        device_str = str(device)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    except Exception:
        pass


def bytes_to_mb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def tensor_list_nbytes(items: Sequence[torch.Tensor]) -> int:
    return int(sum(tensor_nbytes(item) for item in items))


def model_nbytes(module: torch.nn.Module) -> int:
    total = 0
    for parameter in module.parameters():
        total += tensor_nbytes(parameter)
    for buffer in module.buffers():
        total += tensor_nbytes(buffer)
    return int(total)


def build_module_weight_summary(tts: TTS) -> Dict[str, Any]:
    modules = {
        "t2s_model": tts.t2s_model,
        "t2s_core": tts.t2s_model.model if tts.t2s_model is not None else None,
        "vits_model": tts.vits_model,
        "bert_model": tts.bert_model,
        "cnhuhbert_model": tts.cnhuhbert_model,
        "vocoder": tts.vocoder,
        "sv_model": tts.sv_model,
    }
    by_module = {}
    total_bytes = 0
    for name, module in modules.items():
        module_bytes = model_nbytes(module) if module is not None else 0
        by_module[name] = {
            "bytes": int(module_bytes),
            "mb": bytes_to_mb(module_bytes),
        }
        total_bytes += module_bytes
    return {
        "by_module": by_module,
        "total_bytes": int(total_bytes),
        "total_mb": bytes_to_mb(total_bytes),
    }


def snapshot_live_cuda_tensors(top_k: int = 40) -> Dict[str, Any]:
    storages: Dict[int, Dict[str, Any]] = {}
    tensor_views: List[Dict[str, Any]] = []
    for obj in gc.get_objects():
        try:
            tensor = None
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            if tensor is None or not tensor.is_cuda:
                continue
            storage = tensor.untyped_storage()
            storage_ptr = int(storage.data_ptr())
            if storage_ptr not in storages:
                storages[storage_ptr] = {
                    "storage_ptr": storage_ptr,
                    "storage_bytes": int(storage.nbytes()),
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "device": str(tensor.device),
                }
            tensor_views.append(
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": tensor_nbytes(tensor),
                    "device": str(tensor.device),
                }
            )
        except Exception:
            continue
    storage_list = sorted(storages.values(), key=lambda item: item["storage_bytes"], reverse=True)
    tensor_views.sort(key=lambda item: item["bytes"], reverse=True)
    return {
        "unique_storage_count": int(len(storage_list)),
        "unique_storage_total_bytes": int(sum(item["storage_bytes"] for item in storage_list)),
        "unique_storage_total_mb": bytes_to_mb(sum(item["storage_bytes"] for item in storage_list)),
        "top_storages": storage_list[:top_k],
        "top_tensor_views": tensor_views[:top_k],
    }


def build_single_spec(args: argparse.Namespace) -> List[SchedulerRequestSpec]:
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8").strip()
    return [
        SchedulerRequestSpec(
            request_id="req_000",
            ref_audio_path=args.ref_audio,
            prompt_text=args.prompt_text,
            prompt_lang=args.prompt_lang,
            text=text,
            text_lang=args.text_lang,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            early_stop_num=args.early_stop_num,
            ready_step=0,
        )
    ]


def build_auto_specs(args: argparse.Namespace) -> List[SchedulerRequestSpec]:
    wav_paths = sorted(args.auto_wav_dir.glob("*.wav"))[: args.auto_count]
    if len(wav_paths) < args.auto_count:
        raise ValueError(f"auto wav count不足，目录 {args.auto_wav_dir} 只有 {len(wav_paths)} 条 wav")
    text_lines = [line.strip() for line in args.auto_text_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(text_lines) < args.auto_count:
        raise ValueError(f"auto text lines不足，文件 {args.auto_text_file} 只有 {len(text_lines)} 行有效文本")
    specs: List[SchedulerRequestSpec] = []
    for index, wav_path in enumerate(wav_paths):
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists():
            raise FileNotFoundError(f"找不到参考文本 {lab_path}")
        specs.append(
            SchedulerRequestSpec(
                request_id=f"req_{index:03d}",
                ref_audio_path=wav_path,
                prompt_text=lab_path.read_text(encoding="utf-8").strip(),
                prompt_lang="zh",
                text=text_lines[index],
                text_lang=args.text_lang,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                early_stop_num=args.early_stop_num,
                ready_step=0,
            )
        )
    return specs


def load_request_specs(args: argparse.Namespace) -> List[SchedulerRequestSpec]:
    if args.request_manifest is not None:
        payload = json.loads(args.request_manifest.read_text(encoding="utf-8"))
        raw_requests = payload["requests"] if isinstance(payload, dict) else payload
        specs: List[SchedulerRequestSpec] = []
        for index, item in enumerate(raw_requests):
            text = item.get("text")
            text_file = item.get("text_file")
            if text is None and text_file is None:
                raise ValueError(f"request[{index}] must provide text or text_file")
            if text is None:
                text = Path(text_file).read_text(encoding="utf-8").strip()
            specs.append(
                SchedulerRequestSpec(
                    request_id=item.get("request_id", f"req_{index:03d}"),
                    ref_audio_path=Path(item["ref_audio_path"]),
                    prompt_text=item["prompt_text"],
                    prompt_lang=item.get("prompt_lang", "zh"),
                    text=text,
                    text_lang=item.get("text_lang", "zh"),
                    top_k=int(item.get("top_k", args.top_k)),
                    top_p=float(item.get("top_p", args.top_p)),
                    temperature=float(item.get("temperature", args.temperature)),
                    repetition_penalty=float(item.get("repetition_penalty", args.repetition_penalty)),
                    early_stop_num=int(item.get("early_stop_num", args.early_stop_num)),
                    ready_step=int(item.get("ready_step", 0)),
                )
            )
        return specs
    if args.scenario == "single":
        return build_single_spec(args)
    return build_auto_specs(args)


def load_pipeline(config_path: Path) -> TTS:
    tts_config = TTS_Config(str(config_path))
    print(tts_config)
    return TTS(tts_config)


def cuda_mem_snapshot(device: Any) -> Dict[str, float]:
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }
    _sync_device(device)
    return {
        "allocated_mb": bytes_to_mb(torch.cuda.memory_allocated(device)),
        "reserved_mb": bytes_to_mb(torch.cuda.memory_reserved(device)),
        "max_allocated_mb": bytes_to_mb(torch.cuda.max_memory_allocated(device)),
        "max_reserved_mb": bytes_to_mb(torch.cuda.max_memory_reserved(device)),
    }


def stage_run(device: Any, fn) -> Tuple[Any, Dict[str, float]]:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        gc.collect()
        _sync_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    before = cuda_mem_snapshot(device)
    started = time.perf_counter()
    result = fn()
    _sync_device(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    after = cuda_mem_snapshot(device)
    after["elapsed_ms"] = float(elapsed_ms)
    after["delta_allocated_mb"] = float(after["allocated_mb"] - before["allocated_mb"])
    after["delta_reserved_mb"] = float(after["reserved_mb"] - before["reserved_mb"])
    after["stage_peak_over_before_mb"] = float(max(after["max_allocated_mb"] - before["allocated_mb"], 0.0))
    return result, after


class GlobalPeakRecorder:
    def __init__(self, device: Any):
        self.device = device
        self.checkpoints: List[Dict[str, Any]] = []
        if str(device).startswith("cuda") and torch.cuda.is_available():
            gc.collect()
            _sync_device(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

    def record(self, label: str, **extra: Any) -> None:
        snapshot = cuda_mem_snapshot(self.device)
        snapshot["label"] = label
        snapshot.update(extra)
        self.checkpoints.append(snapshot)

    def summary(self) -> Dict[str, Any]:
        peak = max(self.checkpoints, key=lambda item: item["max_allocated_mb"]) if self.checkpoints else None
        return {
            "peak_allocated_mb": 0.0 if peak is None else float(peak["max_allocated_mb"]),
            "peak_reserved_mb": 0.0 if peak is None else float(peak["max_reserved_mb"]),
            "peak_label": None if peak is None else peak["label"],
            "checkpoints": self.checkpoints,
        }


def summarise_state_tensors(states: Sequence[T2SRequestState]) -> Dict[str, Any]:
    per_request = []
    total = {
        "phones_bytes": 0,
        "prompt_phones_bytes": 0,
        "all_phones_bytes": 0,
        "all_bert_features_bytes": 0,
        "prompt_semantic_bytes": 0,
        "refer_spec_bytes": 0,
        "raw_audio_bytes": 0,
        "audio_16k_bytes": 0,
    }
    for state in states:
        spec_audio, audio_16k = state.refer_spec
        item = {
            "request_id": state.request_id,
            "prompt_semantic_len": int(state.prompt_semantic.shape[0]),
            "phones_len": int(state.phones.shape[0]),
            "all_phones_len": int(state.all_phones.shape[0]),
            "bert_frames": int(state.all_bert_features.shape[-1]),
            "phones_bytes": tensor_nbytes(state.phones),
            "prompt_phones_bytes": tensor_nbytes(state.prompt_phones),
            "all_phones_bytes": tensor_nbytes(state.all_phones),
            "all_bert_features_bytes": tensor_nbytes(state.all_bert_features),
            "prompt_semantic_bytes": tensor_nbytes(state.prompt_semantic),
            "refer_spec_bytes": tensor_nbytes(spec_audio),
            "audio_16k_bytes": tensor_nbytes(audio_16k),
            "raw_audio_bytes": tensor_nbytes(state.raw_audio),
        }
        for key in total:
            total[key] += int(item[key])
        per_request.append(item)
    total["total_bytes"] = int(sum(total.values()))
    total["total_mb"] = bytes_to_mb(total["total_bytes"])
    return {"per_request": per_request, "total": total}


def summarise_prefill_batch(active_batch: Any) -> Dict[str, Any]:
    y_sequence_bytes = int(sum(tensor_nbytes(item) for item in active_batch.y_sequences))
    fields = {
        "x_bytes": tensor_nbytes(active_batch.x),
        "x_lens_bytes": tensor_nbytes(active_batch.x_lens),
        "prefix_lens_bytes": tensor_nbytes(active_batch.prefix_lens),
        "xy_pos_bytes": tensor_nbytes(active_batch.xy_pos),
        "key_padding_mask_bytes": tensor_nbytes(active_batch.key_padding_mask),
        "prefill_attn_mask_bytes": tensor_nbytes(active_batch.prefill_attn_mask),
        "y_sequence_bytes": y_sequence_bytes,
    }
    fields["total_bytes"] = int(sum(fields.values()))
    fields["total_mb"] = bytes_to_mb(fields["total_bytes"])
    fields["batch_size"] = int(len(active_batch.states))
    fields["max_x_len"] = int(active_batch.x.shape[1])
    fields["src_len"] = int(active_batch.xy_pos.shape[1])
    fields["prefill_attn_mask_shape"] = list(active_batch.prefill_attn_mask.shape)
    return fields


def summarise_running_requests(running_requests: Sequence[T2SRunningRequest]) -> Dict[str, Any]:
    per_request = []
    total_private_k_bytes = 0
    total_private_v_bytes = 0
    total_decode_mask_bytes = 0
    total_y_sequence_bytes = 0
    for item in running_requests:
        k_bytes = tensor_list_nbytes(item.k_cache)
        v_bytes = tensor_list_nbytes(item.v_cache)
        mask_bytes = tensor_nbytes(item.decode_attn_mask)
        y_bytes = tensor_nbytes(item.y_sequence)
        total_private_k_bytes += k_bytes
        total_private_v_bytes += v_bytes
        total_decode_mask_bytes += mask_bytes
        total_y_sequence_bytes += y_bytes
        per_request.append(
            {
                "request_id": item.state.request_id,
                "step_idx": int(item.step_idx),
                "prefix_len": int(item.prefix_len),
                "history_len": int(item.y_sequence.shape[0]),
                "kv_len": int(item.k_cache[0].shape[1]),
                "k_cache_bytes": k_bytes,
                "v_cache_bytes": v_bytes,
                "decode_mask_bytes": mask_bytes,
                "y_sequence_bytes": y_bytes,
            }
        )
    total_bytes = total_private_k_bytes + total_private_v_bytes + total_decode_mask_bytes + total_y_sequence_bytes
    return {
        "per_request": per_request,
        "totals": {
            "private_k_cache_bytes": int(total_private_k_bytes),
            "private_v_cache_bytes": int(total_private_v_bytes),
            "private_kv_cache_bytes": int(total_private_k_bytes + total_private_v_bytes),
            "decode_mask_bytes": int(total_decode_mask_bytes),
            "y_sequence_bytes": int(total_y_sequence_bytes),
            "total_bytes": int(total_bytes),
            "total_mb": bytes_to_mb(total_bytes),
        },
    }


def summarise_decode_batch(
    xy_pos: torch.Tensor,
    batched_k_cache: Sequence[torch.Tensor],
    batched_v_cache: Sequence[torch.Tensor],
    batched_decode_attn_mask: Optional[torch.Tensor],
    running_requests: Sequence[T2SRunningRequest],
) -> Dict[str, Any]:
    private_k_bytes = int(sum(tensor_list_nbytes(item.k_cache) for item in running_requests))
    private_v_bytes = int(sum(tensor_list_nbytes(item.v_cache) for item in running_requests))
    batched_k_bytes = tensor_list_nbytes(batched_k_cache)
    batched_v_bytes = tensor_list_nbytes(batched_v_cache)
    batched_mask_bytes = tensor_nbytes(batched_decode_attn_mask)
    xy_pos_bytes = tensor_nbytes(xy_pos)
    total_bytes = batched_k_bytes + batched_v_bytes + batched_mask_bytes + xy_pos_bytes
    return {
        "batch_size": int(len(running_requests)),
        "xy_pos_bytes": int(xy_pos_bytes),
        "batched_k_cache_bytes": int(batched_k_bytes),
        "batched_v_cache_bytes": int(batched_v_bytes),
        "batched_kv_cache_bytes": int(batched_k_bytes + batched_v_bytes),
        "batched_decode_mask_bytes": int(batched_mask_bytes),
        "private_kv_cache_bytes_reference": int(private_k_bytes + private_v_bytes),
        "kv_padding_overhead_bytes": int((batched_k_bytes + batched_v_bytes) - (private_k_bytes + private_v_bytes)),
        "total_bytes": int(total_bytes),
        "total_mb": bytes_to_mb(total_bytes),
        "xy_pos_shape": list(xy_pos.shape),
        "batched_decode_mask_shape": None if batched_decode_attn_mask is None else list(batched_decode_attn_mask.shape),
        "layer_k_cache_shape": list(batched_k_cache[0].shape),
    }


def summarise_decode_outputs(
    xy_dec: torch.Tensor,
    next_k_cache: Sequence[torch.Tensor],
    next_v_cache: Sequence[torch.Tensor],
) -> Dict[str, Any]:
    xy_dec_bytes = tensor_nbytes(xy_dec)
    next_k_bytes = tensor_list_nbytes(next_k_cache)
    next_v_bytes = tensor_list_nbytes(next_v_cache)
    total_bytes = xy_dec_bytes + next_k_bytes + next_v_bytes
    return {
        "xy_dec_bytes": int(xy_dec_bytes),
        "next_k_cache_bytes": int(next_k_bytes),
        "next_v_cache_bytes": int(next_v_bytes),
        "next_kv_cache_bytes": int(next_k_bytes + next_v_bytes),
        "total_bytes": int(total_bytes),
        "total_mb": bytes_to_mb(total_bytes),
        "xy_dec_shape": list(xy_dec.shape),
        "layer_next_k_cache_shape": list(next_k_cache[0].shape),
    }


def top_rankings(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranking = [
        ("request_state_total", summary["prepare_stage"]["request_state"]["total"]["total_bytes"]),
        ("prefill_batch_total", summary["prefill_batch"]["tensor_bytes"]["total_bytes"]),
        ("running_private_kv", summary["prefill_step"]["running_requests"]["totals"]["private_kv_cache_bytes"]),
        ("decode_batched_kv", summary["decode_batch"]["tensor_bytes"]["batched_kv_cache_bytes"]),
        ("decode_kv_padding_overhead", summary["decode_batch"]["tensor_bytes"]["kv_padding_overhead_bytes"]),
        ("decode_outputs_next_kv", summary["decode_outputs"]["tensor_bytes"]["next_kv_cache_bytes"]),
        ("prefill_attn_mask", summary["prefill_batch"]["tensor_bytes"]["prefill_attn_mask_bytes"]),
    ]
    ranking.sort(key=lambda item: item[1], reverse=True)
    return [{"name": name, "bytes": int(value), "mb": bytes_to_mb(int(value))} for name, value in ranking]


def synthesize_finished_item(tts: TTS, state: T2SRequestState, semantic_tokens: torch.Tensor) -> Tuple[int, np.ndarray]:
    semantic_tokens = semantic_tokens.unsqueeze(0).unsqueeze(0).to(tts.configs.device)
    phones = state.phones.unsqueeze(0).to(tts.configs.device)
    audio_fragment = tts.synthesize_audio_request_local(
        semantic_tokens=semantic_tokens,
        phones=phones,
        prompt_semantic=state.prompt_semantic,
        prompt_phones=state.prompt_phones,
        refer_spec=state.refer_spec,
        raw_audio=state.raw_audio,
        raw_sr=state.raw_sr,
        speed=1.0,
        sample_steps=32,
    )
    output_sr = tts.configs.sampling_rate if not tts.configs.use_vocoder else tts.vocoder_configs["sr"]
    return tts.audio_postprocess(
        audio=[[audio_fragment]],
        sr=int(output_sr),
        batch_index_list=None,
        speed_factor=1.0,
        split_bucket=False,
        fragment_interval=0.0,
        super_sampling=False,
    )


def simulate_worker_end_to_end(
    tts: TTS,
    specs: Sequence[SchedulerRequestSpec],
    max_steps: int,
    rounds: int,
    grad_mode: str = "default",
) -> Dict[str, Any]:
    device = tts.configs.device
    recorder = GlobalPeakRecorder(device)
    recorder.record("after_model_load")

    state_map: Dict[str, T2SRequestState] = {}
    per_round: List[Dict[str, Any]] = []

    for round_index in range(rounds):
        grad_context = torch.inference_mode if grad_mode == "inference_mode" else contextlib.nullcontext
        with grad_context():
            states = [prepare_request_state(tts, spec) for spec in specs]
        state_map = {state.request_id: state for state in states}
        recorder.record(
            "after_prepare_states",
            round_index=int(round_index),
            request_count=int(len(states)),
            grad_mode=grad_mode,
        )

        pending = list(states)
        running_requests: List[T2SRunningRequest] = []
        round_events: List[Dict[str, Any]] = []
        current_tick = 0

        while pending or running_requests:
            admitted = pending
            pending = []

            if admitted:
                recorder.record(
                    "before_prefill",
                    round_index=int(round_index),
                    tick=int(current_tick),
                    admitted_count=int(len(admitted)),
                    running_count=int(len(running_requests)),
                    grad_mode=grad_mode,
                )
                with grad_context():
                    admitted_running, admitted_finished = run_prefill_step(tts.t2s_model.model, admitted, max_steps=max_steps)
                recorder.record(
                    "after_prefill",
                    round_index=int(round_index),
                    tick=int(current_tick),
                    admitted_running_count=int(len(admitted_running)),
                    admitted_finished_count=int(len(admitted_finished)),
                    running_count=int(len(running_requests)),
                    grad_mode=grad_mode,
                )
                round_events.append(
                    {
                        "tick": int(current_tick),
                        "event": "prefill",
                        "admitted_count": int(len(admitted)),
                        "admitted_running_count": int(len(admitted_running)),
                        "admitted_finished_count": int(len(admitted_finished)),
                    }
                )
                for item in admitted_finished:
                    recorder.record(
                        "before_synth_prefill_finished",
                        round_index=int(round_index),
                        tick=int(current_tick),
                        running_count=int(len(running_requests)),
                        finished_request_id=item.request_id,
                        semantic_len=int(item.semantic_tokens.shape[0]),
                        grad_mode=grad_mode,
                    )
                    with grad_context():
                        sample_rate, audio_data = synthesize_finished_item(tts, state_map[item.request_id], item.semantic_tokens)
                    recorder.record(
                        "after_synth_prefill_finished",
                        round_index=int(round_index),
                        tick=int(current_tick),
                        running_count=int(len(running_requests)),
                        finished_request_id=item.request_id,
                        sample_rate=int(sample_rate),
                        audio_samples=int(audio_data.shape[0]),
                        grad_mode=grad_mode,
                    )
                running_requests.extend(admitted_running)
                recorder.record(
                    "after_extend_running",
                    round_index=int(round_index),
                    tick=int(current_tick),
                    running_count=int(len(running_requests)),
                    grad_mode=grad_mode,
                )

            if running_requests:
                recorder.record(
                    "before_decode",
                    round_index=int(round_index),
                    tick=int(current_tick),
                    running_count=int(len(running_requests)),
                    grad_mode=grad_mode,
                )
                with grad_context():
                    running_requests, step_finished = run_decode_step_for_running(
                        tts.t2s_model.model,
                        running_requests,
                        max_steps=max_steps,
                    )
                recorder.record(
                    "after_decode",
                    round_index=int(round_index),
                    tick=int(current_tick),
                    running_count=int(len(running_requests)),
                    finished_count=int(len(step_finished)),
                    grad_mode=grad_mode,
                )
                round_events.append(
                    {
                        "tick": int(current_tick),
                        "event": "decode",
                        "running_count_after_decode": int(len(running_requests)),
                        "finished_count": int(len(step_finished)),
                    }
                )
                for item in step_finished:
                    recorder.record(
                        "before_synth_decode_finished",
                        round_index=int(round_index),
                        tick=int(current_tick),
                        running_count=int(len(running_requests)),
                        finished_request_id=item.request_id,
                        semantic_len=int(item.semantic_tokens.shape[0]),
                        grad_mode=grad_mode,
                    )
                    with grad_context():
                        sample_rate, audio_data = synthesize_finished_item(tts, state_map[item.request_id], item.semantic_tokens)
                    recorder.record(
                        "after_synth_decode_finished",
                        round_index=int(round_index),
                        tick=int(current_tick),
                        running_count=int(len(running_requests)),
                        finished_request_id=item.request_id,
                        sample_rate=int(sample_rate),
                        audio_samples=int(audio_data.shape[0]),
                        grad_mode=grad_mode,
                    )
            current_tick += 1

        recorder.record(
            "after_round_complete",
            round_index=int(round_index),
            running_count=0,
            grad_mode=grad_mode,
        )
        per_round.append(
            {
                "round_index": int(round_index),
                "events": round_events,
            }
        )

    return {
        "grad_mode": grad_mode,
        "rounds": per_round,
        "timeline": recorder.summary(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tts = load_pipeline(args.config)
    model = tts.t2s_model.model
    device = tts.configs.device
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    set_seed(args.seed, use_cuda)

    specs = load_request_specs(args)
    if args.early_stop_num == -1:
        for spec in specs:
            spec.early_stop_num = int(tts.configs.hz * tts.configs.max_sec)

    if args.warmup and specs:
        warmup_spec = specs[:1]
        _ = [prepare_request_state(tts, spec) for spec in warmup_spec]
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
            _sync_device(device)

    states, prepare_mem = stage_run(device, lambda: [prepare_request_state(tts, spec) for spec in specs])
    request_state_summary = summarise_state_tensors(states)

    active_batch, prefill_batch_mem = stage_run(device, lambda: build_prefill_batch(model, states))
    prefill_batch_tensor_summary = summarise_prefill_batch(active_batch)

    prefill_result, prefill_step_mem = stage_run(device, lambda: run_prefill_step(model, states, max_steps=args.max_steps))
    running_requests, finished_items = prefill_result
    running_requests_summary = summarise_running_requests(running_requests)
    finished_after_prefill_summary = [
        {
            "request_id": item.request_id,
            "finish_idx": int(item.finish_idx),
            "finish_reason": item.finish_reason,
            "semantic_len": int(item.semantic_tokens.shape[0]),
        }
        for item in finished_items
    ]

    if not running_requests:
        raise RuntimeError(f"prefill 后没有 running requests，全部在首步结束: {[item.request_id for item in finished_items]}")

    decode_batch_result, decode_batch_mem = stage_run(
        device,
        lambda: _build_decode_batch_from_running(model, running_requests),
    )
    xy_pos, batched_k_cache, batched_v_cache, batched_decode_attn_mask = decode_batch_result
    decode_batch_tensor_summary = summarise_decode_batch(
        xy_pos,
        batched_k_cache,
        batched_v_cache,
        batched_decode_attn_mask,
        running_requests,
    )

    decode_out_result, decode_step_mem = stage_run(
        device,
        lambda: model.t2s_transformer.decode_next_token(
            xy_pos,
            batched_k_cache,
            batched_v_cache,
            batched_decode_attn_mask,
        ),
    )
    xy_dec, next_k_cache, next_v_cache = decode_out_result
    decode_output_tensor_summary = summarise_decode_outputs(xy_dec, next_k_cache, next_v_cache)
    del active_batch
    del running_requests
    del finished_items
    del xy_pos
    del batched_k_cache
    del batched_v_cache
    del batched_decode_attn_mask
    del xy_dec
    del next_k_cache
    del next_v_cache
    gc.collect()
    if use_cuda:
        _sync_device(device)
        torch.cuda.empty_cache()
    end_to_end_worker = simulate_worker_end_to_end(
        tts=tts,
        specs=specs,
        max_steps=args.max_steps,
        rounds=args.worker_rounds,
        grad_mode=args.worker_grad_mode,
    )
    live_cuda_tensors_after_worker = snapshot_live_cuda_tensors()
    worker_inference_mode = None
    if args.compare_worker_grad_modes:
        gc.collect()
        if use_cuda:
            _sync_device(device)
            torch.cuda.empty_cache()
        worker_inference_mode = simulate_worker_end_to_end(
            tts=tts,
            specs=specs,
            max_steps=args.max_steps,
            rounds=args.worker_rounds,
            grad_mode="inference_mode",
        )

    summary = {
        "meta": {
            "scenario": args.scenario if args.request_manifest is None else "manifest",
            "seed": int(args.seed),
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "request_count": int(len(specs)),
            "num_layers": int(model.num_layers),
            "num_heads": int(model.num_head),
            "model_dim": int(model.model_dim),
            "model_weights_mb": bytes_to_mb(model_nbytes(model)),
        },
        "loaded_module_weights": build_module_weight_summary(tts),
        "requests": [
            {
                "request_id": spec.request_id,
                "ref_audio_path": str(spec.ref_audio_path),
                "prompt_text": spec.prompt_text,
                "text": spec.text,
            }
            for spec in specs
        ],
        "prepare_stage": {
            "memory": prepare_mem,
            "request_state": request_state_summary,
        },
        "prefill_batch": {
            "memory": prefill_batch_mem,
            "tensor_bytes": prefill_batch_tensor_summary,
        },
        "prefill_step": {
            "memory": prefill_step_mem,
            "running_requests": running_requests_summary,
            "finished_after_prefill": finished_after_prefill_summary,
        },
        "decode_batch": {
            "memory": decode_batch_mem,
            "tensor_bytes": decode_batch_tensor_summary,
        },
        "decode_outputs": {
            "memory": decode_step_mem,
            "tensor_bytes": decode_output_tensor_summary,
        },
        "end_to_end_worker": end_to_end_worker,
        "live_cuda_tensors_after_worker": live_cuda_tensors_after_worker,
        "end_to_end_worker_inference_mode": worker_inference_mode,
    }
    summary["top_rankings"] = top_rankings(summary)

    summary_path = args.output_dir / "t2s_memory_breakdown_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary["meta"], ensure_ascii=False, indent=2))
    print("[top_rankings]")
    for item in summary["top_rankings"]:
        print(f"- {item['name']}: {item['mb']:.3f} MB")
    print("[worker_peak]")
    print(
        json.dumps(
            {
                "peak_label": summary["end_to_end_worker"]["timeline"]["peak_label"],
                "peak_allocated_mb": summary["end_to_end_worker"]["timeline"]["peak_allocated_mb"],
                "peak_reserved_mb": summary["end_to_end_worker"]["timeline"]["peak_reserved_mb"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if worker_inference_mode is not None:
        print("[worker_peak_inference_mode]")
        print(
            json.dumps(
                {
                    "peak_label": worker_inference_mode["timeline"]["peak_label"],
                    "peak_allocated_mb": worker_inference_mode["timeline"]["peak_allocated_mb"],
                    "peak_reserved_mb": worker_inference_mode["timeline"]["peak_reserved_mb"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
