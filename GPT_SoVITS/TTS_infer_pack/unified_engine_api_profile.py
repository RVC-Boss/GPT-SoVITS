from __future__ import annotations

from typing import Any, Dict, List, Sequence

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SFinishedItem, T2SRequestState


def build_request_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text")
    prompt_text = payload.get("prompt_text")
    return {
        "text_len": 0 if text is None else len(str(text)),
        "prompt_text_len": 0 if prompt_text is None else len(str(prompt_text)),
        "text_lang": payload.get("text_lang"),
        "prompt_lang": payload.get("prompt_lang"),
        "ref_audio_path": payload.get("ref_audio_path"),
    }


def sum_profile_field(items: Sequence[Dict[str, Any]], key: str) -> float:
    total = 0.0
    for item in items:
        value = item.get(key, 0.0)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def aggregate_numeric_dicts(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def build_direct_segment_trace(
    segment_texts: Sequence[str],
    prepare_profiles: Sequence[Dict[str, Any]],
    worker_profiles: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for index, segment_text in enumerate(segment_texts):
        prepare_item = prepare_profiles[index] if index < len(prepare_profiles) else {}
        worker_item = worker_profiles[index] if index < len(worker_profiles) else {}
        prepare_profile = dict(prepare_item.get("prepare_profile", {}))
        results.append(
            {
                "segment_index": index,
                "request_id": prepare_item.get("request_id") or worker_item.get("request_id"),
                "text_len": len(str(segment_text)),
                "prepare_wall_ms": float(prepare_item.get("prepare_wall_ms", 0.0)),
                "prepare_profile_total_ms": float(prepare_item.get("prepare_profile_total_ms", 0.0)),
                "prepare_engine_gpu_queue_wait_ms": float(
                    dict(prepare_item.get("prepare_profile", {})).get("engine_gpu_prepare_queue_wait_ms", 0.0)
                ),
                "engine_policy_wait_ms": float(prepare_item.get("engine_policy_wait_ms", 0.0)),
                "engine_dispatch_wait_ms": float(prepare_item.get("engine_dispatch_wait_ms", 0.0)),
                "decode_admission_wait_ms": float(worker_item.get("decode_admission_wait_ms", 0.0)),
                "queue_wait_ms": float(worker_item.get("queue_wait_ms", 0.0)),
                "prefill_ms": float(worker_item.get("prefill_ms", 0.0)),
                "merge_ms": float(worker_item.get("merge_ms", 0.0)),
                "decode_ms": float(worker_item.get("decode_ms", 0.0)),
                "finalize_wait_ms": float(worker_item.get("finalize_wait_ms", 0.0)),
                "synth_ms": float(worker_item.get("synth_ms", 0.0)),
                "worker_total_ms": float(worker_item.get("worker_total_ms", 0.0)),
                "decode_steps": int(worker_item.get("decode_steps", 0)),
                "semantic_len": int(worker_item.get("semantic_len", 0)),
                "finish_reason": worker_item.get("finish_reason"),
                "norm_text": prepare_profile.get("norm_text"),
            }
        )
    return results


def build_direct_scheduler_profile(
    *,
    backend: str,
    request_start: float,
    response_ready_at: float,
    audio_bytes: int,
    sample_rate: int,
    segment_texts: Sequence[str],
    prepare_profiles: Sequence[Dict[str, Any]],
    worker_profiles: Sequence[Dict[str, Any]],
    pack_ms: float,
    response_overhead_ms: float,
) -> Dict[str, Any]:
    segment_trace = build_direct_segment_trace(segment_texts, prepare_profiles, worker_profiles)
    prepare_profile_dicts = [dict(item.get("prepare_profile", {})) for item in prepare_profiles]
    request_total_ms = max(0.0, (response_ready_at - request_start) * 1000.0)
    prepare_wall_ms = sum_profile_field(prepare_profiles, "prepare_wall_ms")
    prepare_profile_total_ms = sum_profile_field(prepare_profiles, "prepare_profile_total_ms")
    engine_policy_wait_ms = sum_profile_field(prepare_profiles, "engine_policy_wait_ms")
    engine_dispatch_wait_ms = sum_profile_field(prepare_profiles, "engine_dispatch_wait_ms")
    decode_admission_wait_ms = sum_profile_field(worker_profiles, "decode_admission_wait_ms")
    queue_wait_ms = sum_profile_field(worker_profiles, "queue_wait_ms")
    prefill_ms = sum_profile_field(worker_profiles, "prefill_ms")
    merge_ms = sum_profile_field(worker_profiles, "merge_ms")
    decode_ms = sum_profile_field(worker_profiles, "decode_ms")
    finalize_wait_ms = sum_profile_field(worker_profiles, "finalize_wait_ms")
    synth_ms = sum_profile_field(worker_profiles, "synth_ms")
    worker_total_ms = sum_profile_field(worker_profiles, "worker_total_ms")
    decode_steps = sum(int(item.get("decode_steps", 0)) for item in worker_profiles)
    semantic_len = sum(int(item.get("semantic_len", 0)) for item in worker_profiles)
    request_other_ms = max(
        0.0,
        request_total_ms - prepare_wall_ms - engine_policy_wait_ms - worker_total_ms - pack_ms - response_overhead_ms,
    )
    return {
        "backend": backend,
        "backend_mode": backend,
        "segment_count": len(segment_texts),
        "sample_rate": int(sample_rate),
        "audio_bytes": int(audio_bytes),
        "request_total_ms": request_total_ms,
        "prepare_ms": prepare_wall_ms,
        "prepare_wall_ms": prepare_wall_ms,
        "prepare_profile_total_ms": prepare_profile_total_ms,
        "engine_policy_wait_ms": engine_policy_wait_ms,
        "engine_dispatch_wait_ms": engine_dispatch_wait_ms,
        "decode_admission_wait_ms": decode_admission_wait_ms,
        "queue_wait_ms": queue_wait_ms,
        "prefill_ms": prefill_ms,
        "merge_ms": merge_ms,
        "decode_ms": decode_ms,
        "finalize_wait_ms": finalize_wait_ms,
        "synth_ms": synth_ms,
        "pack_ms": pack_ms,
        "response_overhead_ms": response_overhead_ms,
        "worker_total_ms": worker_total_ms,
        "request_other_ms": request_other_ms,
        "decode_steps": decode_steps,
        "semantic_len": semantic_len,
        "prepare_segments": list(prepare_profiles),
        "worker_segments": list(worker_profiles),
        "segment_trace": segment_trace,
        "prepare_aggregate": aggregate_numeric_dicts(prepare_profile_dicts),
    }


def build_legacy_direct_profile(
    *,
    backend: str,
    fallback_reason: str | None,
    request_start: float,
    finished_at: float,
    sample_rate: int | None = None,
    audio_bytes: int = 0,
    pack_ms: float = 0.0,
    chunk_count: int = 0,
    stream_total_bytes: int = 0,
    first_chunk_ms: float | None = None,
) -> Dict[str, Any]:
    request_total_ms = max(0.0, (finished_at - request_start) * 1000.0)
    legacy_infer_ms = max(0.0, request_total_ms - pack_ms)
    return {
        "backend": backend,
        "backend_mode": backend,
        "fallback_reason": fallback_reason,
        "request_total_ms": request_total_ms,
        "prepare_ms": 0.0,
        "queue_wait_ms": 0.0,
        "prefill_ms": 0.0,
        "merge_ms": 0.0,
        "decode_ms": 0.0,
        "finalize_wait_ms": 0.0,
        "synth_ms": 0.0,
        "pack_ms": pack_ms,
        "worker_total_ms": legacy_infer_ms,
        "request_other_ms": 0.0,
        "legacy_infer_ms": legacy_infer_ms,
        "sample_rate": int(sample_rate) if sample_rate is not None else None,
        "audio_bytes": int(audio_bytes),
        "chunk_count": int(chunk_count),
        "stream_total_bytes": int(stream_total_bytes),
        "first_chunk_ms": None if first_chunk_ms is None else float(first_chunk_ms),
    }


def build_scheduler_submit_profile(
    *,
    backend: str,
    request_start: float,
    response_ready_at: float,
    audio_bytes: int,
    sample_rate: int,
    prepare_spec_build_ms: float,
    prepare_wall_ms: float,
    prepare_executor_queue_ms: float,
    prepare_executor_run_ms: float,
    prepare_profile_total_ms: float,
    prepare_profile_wall_ms: float,
    prepare_other_ms: float,
    engine_policy_wait_ms: float,
    api_after_prepare_ms: float,
    api_wait_result_ms: float,
    pack_ms: float,
    response_overhead_ms: float,
    worker_profile: Dict[str, Any],
) -> Dict[str, Any]:
    worker_total_ms = float(worker_profile.get("worker_total_ms", 0.0))
    request_total_ms = max(0.0, (response_ready_at - request_start) * 1000.0)
    request_other_ms = max(
        0.0,
        request_total_ms
        - prepare_wall_ms
        - engine_policy_wait_ms
        - api_after_prepare_ms
        - worker_total_ms
        - api_wait_result_ms
        - pack_ms,
    )
    result = {
        "backend": backend,
        "backend_mode": backend,
        "audio_bytes": int(audio_bytes),
        "sample_rate": int(sample_rate),
        "prepare_spec_build_ms": prepare_spec_build_ms,
        "prepare_ms": prepare_wall_ms,
        "prepare_wall_ms": prepare_wall_ms,
        "prepare_executor_queue_ms": prepare_executor_queue_ms,
        "prepare_executor_run_ms": prepare_executor_run_ms,
        "prepare_profile_total_ms": prepare_profile_total_ms,
        "prepare_profile_wall_ms": prepare_profile_wall_ms,
        "prepare_other_ms": prepare_other_ms,
        "engine_policy_wait_ms": float(engine_policy_wait_ms),
        "api_after_prepare_ms": api_after_prepare_ms,
        "api_wait_result_ms": api_wait_result_ms,
        "pack_ms": pack_ms,
        "response_overhead_ms": response_overhead_ms,
        "request_total_ms": request_total_ms,
        "request_other_ms": request_other_ms,
    }
    result.update({key: value for key, value in worker_profile.items()})
    return result


def format_ms_header(value: Any) -> str:
    return f"{float(value):.3f}"


def build_scheduler_submit_headers(
    *,
    request_id: str,
    media_type: str,
    sample_rate: int,
    profile: Dict[str, Any],
) -> Dict[str, str]:
    prepare_profile = dict(profile.get("prepare_profile", {}))
    headers = {
        "X-Request-Id": request_id,
        "X-Semantic-Len": str(int(profile.get("semantic_len", 0))),
        "X-Finish-Reason": str(profile.get("finish_reason", "unknown")),
        "X-Queue-Wait-Ms": format_ms_header(profile.get("queue_wait_ms", 0.0)),
        "X-Decode-Admission-Wait-Ms": format_ms_header(profile.get("decode_admission_wait_ms", 0.0)),
        "X-Engine-Policy-Wait-Ms": format_ms_header(profile.get("engine_policy_wait_ms", 0.0)),
        "X-Engine-Dispatch-Wait-Ms": format_ms_header(profile.get("engine_dispatch_wait_ms", 0.0)),
        "X-Prepare-Ms": format_ms_header(profile.get("prepare_wall_ms", 0.0)),
        "X-Prepare-Wall-Ms": format_ms_header(profile.get("prepare_wall_ms", 0.0)),
        "X-Prepare-Spec-Build-Ms": format_ms_header(profile.get("prepare_spec_build_ms", 0.0)),
        "X-Prepare-Executor-Queue-Ms": format_ms_header(profile.get("prepare_executor_queue_ms", 0.0)),
        "X-Prepare-Admission-Wait-Ms": format_ms_header(prepare_profile.get("prepare_admission_wait_ms", 0.0)),
        "X-Prepare-Executor-Run-Ms": format_ms_header(profile.get("prepare_executor_run_ms", 0.0)),
        "X-Prepare-Profile-Total-Ms": format_ms_header(profile.get("prepare_profile_total_ms", 0.0)),
        "X-Prepare-Profile-Wall-Ms": format_ms_header(profile.get("prepare_profile_wall_ms", 0.0)),
        "X-Prepare-Other-Ms": format_ms_header(profile.get("prepare_other_ms", 0.0)),
        "X-Api-After-Prepare-Ms": format_ms_header(profile.get("api_after_prepare_ms", 0.0)),
        "X-Prefill-Ms": format_ms_header(profile.get("prefill_ms", 0.0)),
        "X-Merge-Ms": format_ms_header(profile.get("merge_ms", 0.0)),
        "X-Decode-Ms": format_ms_header(profile.get("decode_ms", 0.0)),
        "X-Finalize-Wait-Ms": format_ms_header(profile.get("finalize_wait_ms", 0.0)),
        "X-Synth-Ms": format_ms_header(profile.get("synth_ms", 0.0)),
        "X-Worker-Residual-Ms": format_ms_header(profile.get("worker_residual_ms", 0.0)),
        "X-Worker-Other-Ms": format_ms_header(profile.get("worker_other_ms", 0.0)),
        "X-Pack-Ms": format_ms_header(profile.get("pack_ms", 0.0)),
        "X-Worker-Total-Ms": format_ms_header(profile.get("worker_total_ms", 0.0)),
        "X-Api-Wait-Result-Ms": format_ms_header(profile.get("api_wait_result_ms", 0.0)),
        "X-Decode-Steps": str(int(profile.get("decode_steps", 0))),
        "X-Sample-Rate": str(int(sample_rate)),
        "X-Response-Overhead-Ms": format_ms_header(profile.get("response_overhead_ms", 0.0)),
        "X-Request-Other-Ms": format_ms_header(profile.get("request_other_ms", 0.0)),
        "X-Request-Total-Ms": format_ms_header(profile.get("request_total_ms", 0.0)),
    }
    headers.update(
        {
            "X-Prepare-Prompt-Text-Ms": format_ms_header(prepare_profile.get("prompt_text_features_ms", 0.0)),
            "X-Prepare-Target-Text-Ms": format_ms_header(prepare_profile.get("text_features_ms", 0.0)),
            "X-Prepare-Prompt-Text-CPU-Preprocess-Ms": format_ms_header(prepare_profile.get("prompt_text_cpu_preprocess_ms", 0.0)),
            "X-Prepare-Target-Text-CPU-Preprocess-Ms": format_ms_header(prepare_profile.get("text_cpu_preprocess_ms", 0.0)),
            "X-Prepare-Prompt-Text-CPU-Queue-Ms": format_ms_header(prepare_profile.get("prompt_text_cpu_queue_ms", 0.0)),
            "X-Prepare-Target-Text-CPU-Queue-Ms": format_ms_header(prepare_profile.get("text_cpu_queue_ms", 0.0)),
            "X-Prepare-Prompt-Text-Feature-Queue-Ms": format_ms_header(prepare_profile.get("prompt_text_feature_queue_ms", 0.0)),
            "X-Prepare-Target-Text-Feature-Queue-Ms": format_ms_header(prepare_profile.get("text_feature_queue_ms", 0.0)),
            "X-Prepare-Prompt-Bert-Wait-Ms": format_ms_header(prepare_profile.get("prompt_text_bert_wait_ms", 0.0)),
            "X-Prepare-Target-Bert-Wait-Ms": format_ms_header(prepare_profile.get("text_bert_wait_ms", 0.0)),
            "X-Prepare-Prompt-Bert-Admission-Wait-Ms": format_ms_header(prepare_profile.get("prompt_text_bert_admission_wait_ms", 0.0)),
            "X-Prepare-Target-Bert-Admission-Wait-Ms": format_ms_header(prepare_profile.get("text_bert_admission_wait_ms", 0.0)),
            "X-Prepare-Prompt-Bert-Queue-Wait-Ms": format_ms_header(prepare_profile.get("prompt_text_bert_queue_wait_ms", 0.0)),
            "X-Prepare-Target-Bert-Queue-Wait-Ms": format_ms_header(prepare_profile.get("text_bert_queue_wait_ms", 0.0)),
            "X-Prepare-Prompt-Bert-Batch-Collect-Wait-Ms": format_ms_header(prepare_profile.get("prompt_text_bert_batch_collect_wait_ms", 0.0)),
            "X-Prepare-Target-Bert-Batch-Collect-Wait-Ms": format_ms_header(prepare_profile.get("text_bert_batch_collect_wait_ms", 0.0)),
            "X-Prepare-Prompt-Bert-Forward-Ms": format_ms_header(prepare_profile.get("prompt_text_bert_forward_ms", 0.0)),
            "X-Prepare-Target-Bert-Forward-Ms": format_ms_header(prepare_profile.get("text_bert_forward_ms", 0.0)),
            "X-Prepare-Prompt-Bert-Pending-On-Enqueue-Peak": str(int(prepare_profile.get("prompt_text_bert_pending_depth_on_enqueue_peak", 0.0))),
            "X-Prepare-Target-Bert-Pending-On-Enqueue-Peak": str(int(prepare_profile.get("text_bert_pending_depth_on_enqueue_peak", 0.0))),
            "X-Prepare-Prompt-Bert-Pending-On-Collect-Peak": str(int(prepare_profile.get("prompt_text_bert_pending_depth_on_collect_peak", 0.0))),
            "X-Prepare-Target-Bert-Pending-On-Collect-Peak": str(int(prepare_profile.get("text_bert_pending_depth_on_collect_peak", 0.0))),
            "X-Prepare-Prompt-Bert-High-Pressure-Peak": str(int(prepare_profile.get("prompt_text_bert_high_pressure_mode_peak", 0.0))),
            "X-Prepare-Target-Bert-High-Pressure-Peak": str(int(prepare_profile.get("text_bert_high_pressure_mode_peak", 0.0))),
            "X-Prepare-Prompt-Bert-Batch-Window-Ms": format_ms_header(prepare_profile.get("prompt_text_bert_batch_window_ms", 0.0)),
            "X-Prepare-Target-Bert-Batch-Window-Ms": format_ms_header(prepare_profile.get("text_bert_batch_window_ms", 0.0)),
            "X-Prepare-Text-Pair-Wall-Ms": format_ms_header(prepare_profile.get("text_feature_pair_ms", 0.0)),
            "X-Prepare-Text-CPU-Workers": str(int(prepare_profile.get("text_cpu_parallel_workers", 0.0))),
            "X-Prepare-Engine-GPU-Queue-Wait-Ms": format_ms_header(prepare_profile.get("engine_gpu_prepare_queue_wait_ms", 0.0)),
            "X-Prepare-Engine-GPU-Batch-Size": str(int(prepare_profile.get("engine_gpu_prepare_batch_size", 0.0))),
            "X-Prepare-Audio-Load-Ms": format_ms_header(prepare_profile.get("audio_load_ms", 0.0)),
            "X-Prepare-Audio-Stage-Wait-Ms": format_ms_header(prepare_profile.get("audio_stage_wait_ms", 0.0)),
            "X-Prepare-Prompt-Semantic-Ms": format_ms_header(prepare_profile.get("prompt_semantic_ms", 0.0)),
            "X-Prepare-Prompt-Semantic-Wait-Ms": format_ms_header(prepare_profile.get("prompt_semantic_wait_ms", 0.0)),
            "X-Prepare-Prompt-Semantic-CPU-Ms": format_ms_header(prepare_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)),
            "X-Prepare-Prompt-Semantic-Forward-Ms": format_ms_header(prepare_profile.get("prompt_semantic_forward_ms", 0.0)),
            "X-Prepare-Ref-Spec-Ms": format_ms_header(prepare_profile.get("ref_spec_ms", 0.0)),
            "X-Prepare-Ref-Spec-Wait-Ms": format_ms_header(prepare_profile.get("ref_spec_wait_ms", 0.0)),
            "X-Prepare-Ref-Bundle-Ms": format_ms_header(prepare_profile.get("ref_audio_bundle_ms", 0.0)),
            "X-Prepare-Tensorize-Ms": format_ms_header(prepare_profile.get("tensorize_ms", 0.0)),
            "X-Prepare-Inflight-On-Enter": str(int(prepare_profile.get("worker_prepare_inflight_on_enter", 0.0))),
            "X-Prepare-Inflight-Peak": str(int(prepare_profile.get("worker_prepare_peak_inflight", 0.0))),
        }
    )
    return headers


def build_scheduler_debug_request_profile(
    *,
    state: T2SRequestState,
    item: T2SFinishedItem,
    batch_request_count: int,
    prepare_batch_wall_ms: float,
    decode_batch_wall_ms: float,
    batch_request_total_ms: float,
) -> Dict[str, Any]:
    prepare_profile = dict(state.prepare_profile)
    prepare_wall_ms = float(prepare_profile.get("wall_total_ms", 0.0))
    return {
        "backend": "scheduler_debug",
        "backend_mode": "scheduler_debug",
        "batch_request_count": int(batch_request_count),
        "batch_prepare_wall_ms": float(prepare_batch_wall_ms),
        "batch_decode_wall_ms": float(decode_batch_wall_ms),
        "batch_request_total_ms": float(batch_request_total_ms),
        "prepare_ms": prepare_wall_ms,
        "prepare_wall_ms": prepare_wall_ms,
        "prepare_profile_total_ms": float(prepare_profile.get("wall_total_ms", prepare_wall_ms)),
        "prepare_profile": prepare_profile,
        "decode_steps": int(item.finish_idx),
        "finish_idx": int(item.finish_idx),
        "semantic_len": int(item.semantic_tokens.shape[0]),
        "finish_reason": item.finish_reason,
        "norm_text": state.norm_text,
        "norm_prompt_text": state.norm_prompt_text,
    }


def build_scheduler_debug_batch_profile(
    *,
    request_count: int,
    max_steps: int,
    prepare_batch_wall_ms: float,
    decode_batch_wall_ms: float,
    request_total_ms: float,
    finished_items: Sequence[T2SFinishedItem],
) -> Dict[str, Any]:
    finish_reason_counts: Dict[str, int] = {}
    total_semantic_len = 0
    for item in finished_items:
        finish_reason_counts[item.finish_reason] = finish_reason_counts.get(item.finish_reason, 0) + 1
        total_semantic_len += int(item.semantic_tokens.shape[0])
    return {
        "request_count": int(request_count),
        "max_steps": int(max_steps),
        "prepare_batch_wall_ms": float(prepare_batch_wall_ms),
        "decode_batch_wall_ms": float(decode_batch_wall_ms),
        "request_total_ms": float(request_total_ms),
        "total_semantic_len": int(total_semantic_len),
        "finish_reason_counts": finish_reason_counts,
    }
