from __future__ import annotations

import asyncio
import time
import uuid
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SFinishedItem, T2SRequestState, run_scheduler_continuous
from GPT_SoVITS.TTS_infer_pack.unified_engine_audio import pack_audio, set_scheduler_seed, wave_header_chunk
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import (
    DirectTTSExecution,
    EngineStatus,
    NormalizedEngineRequest,
    SchedulerDebugExecution,
    SchedulerPendingJob,
    SchedulerSubmitExecution,
)


class EngineApiFacade:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    @property
    def tts(self):
        return self.owner.tts

    @property
    def cut_method_names(self):
        return self.owner.cut_method_names

    @property
    def reference_registry(self):
        return self.owner.reference_registry

    @property
    def direct_tts_lock(self):
        return self.owner.direct_tts_lock

    @property
    def scheduler_worker(self):
        return self.owner.scheduler_worker

    def _register_request_state(
        self,
        request_id: str,
        api_mode: str,
        backend: str,
        media_type: str,
        response_streaming: bool,
        deadline_ts: float | None = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        return self.owner._register_request_state(
            request_id=request_id,
            api_mode=api_mode,
            backend=backend,
            media_type=media_type,
            response_streaming=response_streaming,
            deadline_ts=deadline_ts,
            meta=meta,
        )

    def _update_request_state(
        self,
        request_id: str,
        status: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.owner._update_request_state(request_id, status, extra)

    def _merge_request_state_profile(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.owner._merge_request_state_profile(request_id, extra)

    def _complete_request_state(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.owner._complete_request_state(request_id, extra)

    def _fail_request_state(self, request_id: str, error: str) -> None:
        self.owner._fail_request_state(request_id, error)

    async def _prepare_state_via_engine_gpu_queue(
        self,
        *,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
        engine_request_id: str | None,
    ) -> tuple[T2SRequestState, float, float]:
        return await self.owner._prepare_state_via_engine_gpu_queue(
            spec=spec,
            prepare_submit_at=prepare_submit_at,
            engine_request_id=engine_request_id,
        )

    async def _enqueue_prepared_state_for_dispatch(
        self,
        *,
        state: T2SRequestState,
        speed_factor: float,
        sample_steps: int,
        media_type: str,
        prepare_wall_ms: float,
        prepare_profile_total_ms: float,
        done_loop: asyncio.AbstractEventLoop | None,
        done_future: asyncio.Future | None,
        engine_request_id: str | None,
        timeout_sec: float | None,
    ):
        return await self.owner._enqueue_prepared_state_for_dispatch(
            state=state,
            speed_factor=speed_factor,
            sample_steps=sample_steps,
            media_type=media_type,
            prepare_wall_ms=prepare_wall_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            done_loop=done_loop,
            done_future=done_future,
            engine_request_id=engine_request_id,
            timeout_sec=timeout_sec,
        )

    def _collect_request_summaries(self, request_ids: Sequence[str]) -> List[Dict[str, Any]]:
        return self.owner.request_registry.collect_summaries(request_ids)

    def _has_active_request(self, request_id: str) -> bool:
        return self.owner.request_registry.has_active(request_id)

    @staticmethod
    def _build_request_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text")
        prompt_text = payload.get("prompt_text")
        return {
            "text_len": 0 if text is None else len(str(text)),
            "prompt_text_len": 0 if prompt_text is None else len(str(prompt_text)),
            "text_lang": payload.get("text_lang"),
            "prompt_lang": payload.get("prompt_lang"),
            "ref_audio_path": payload.get("ref_audio_path"),
        }

    @staticmethod
    def _sum_profile_field(items: Sequence[Dict[str, Any]], key: str) -> float:
        total = 0.0
        for item in items:
            value = item.get(key, 0.0)
            if isinstance(value, (int, float)):
                total += float(value)
        return total

    def _build_direct_segment_trace(
        self,
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

    def _build_direct_scheduler_profile(
        self,
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
        segment_trace = self._build_direct_segment_trace(segment_texts, prepare_profiles, worker_profiles)
        prepare_profile_dicts = [dict(item.get("prepare_profile", {})) for item in prepare_profiles]
        request_total_ms = max(0.0, (response_ready_at - request_start) * 1000.0)
        prepare_wall_ms = self._sum_profile_field(prepare_profiles, "prepare_wall_ms")
        prepare_profile_total_ms = self._sum_profile_field(prepare_profiles, "prepare_profile_total_ms")
        engine_policy_wait_ms = self._sum_profile_field(prepare_profiles, "engine_policy_wait_ms")
        engine_dispatch_wait_ms = self._sum_profile_field(prepare_profiles, "engine_dispatch_wait_ms")
        decode_admission_wait_ms = self._sum_profile_field(worker_profiles, "decode_admission_wait_ms")
        queue_wait_ms = self._sum_profile_field(worker_profiles, "queue_wait_ms")
        prefill_ms = self._sum_profile_field(worker_profiles, "prefill_ms")
        merge_ms = self._sum_profile_field(worker_profiles, "merge_ms")
        decode_ms = self._sum_profile_field(worker_profiles, "decode_ms")
        finalize_wait_ms = self._sum_profile_field(worker_profiles, "finalize_wait_ms")
        synth_ms = self._sum_profile_field(worker_profiles, "synth_ms")
        worker_total_ms = self._sum_profile_field(worker_profiles, "worker_total_ms")
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
            "prepare_aggregate": self._aggregate_numeric_dicts(prepare_profile_dicts),
        }

    def _build_legacy_direct_profile(
        self,
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

    def _build_scheduler_submit_profile(
        self,
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

    @staticmethod
    def _format_ms_header(value: Any) -> str:
        return f"{float(value):.3f}"

    def _build_scheduler_submit_headers(
        self,
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
            "X-Queue-Wait-Ms": self._format_ms_header(profile.get("queue_wait_ms", 0.0)),
            "X-Decode-Admission-Wait-Ms": self._format_ms_header(profile.get("decode_admission_wait_ms", 0.0)),
            "X-Engine-Policy-Wait-Ms": self._format_ms_header(profile.get("engine_policy_wait_ms", 0.0)),
            "X-Engine-Dispatch-Wait-Ms": self._format_ms_header(profile.get("engine_dispatch_wait_ms", 0.0)),
            "X-Prepare-Ms": self._format_ms_header(profile.get("prepare_wall_ms", 0.0)),
            "X-Prepare-Wall-Ms": self._format_ms_header(profile.get("prepare_wall_ms", 0.0)),
            "X-Prepare-Spec-Build-Ms": self._format_ms_header(profile.get("prepare_spec_build_ms", 0.0)),
            "X-Prepare-Executor-Queue-Ms": self._format_ms_header(profile.get("prepare_executor_queue_ms", 0.0)),
            "X-Prepare-Admission-Wait-Ms": self._format_ms_header(prepare_profile.get("prepare_admission_wait_ms", 0.0)),
            "X-Prepare-Executor-Run-Ms": self._format_ms_header(profile.get("prepare_executor_run_ms", 0.0)),
            "X-Prepare-Profile-Total-Ms": self._format_ms_header(profile.get("prepare_profile_total_ms", 0.0)),
            "X-Prepare-Profile-Wall-Ms": self._format_ms_header(profile.get("prepare_profile_wall_ms", 0.0)),
            "X-Prepare-Other-Ms": self._format_ms_header(profile.get("prepare_other_ms", 0.0)),
            "X-Api-After-Prepare-Ms": self._format_ms_header(profile.get("api_after_prepare_ms", 0.0)),
            "X-Prefill-Ms": self._format_ms_header(profile.get("prefill_ms", 0.0)),
            "X-Merge-Ms": self._format_ms_header(profile.get("merge_ms", 0.0)),
            "X-Decode-Ms": self._format_ms_header(profile.get("decode_ms", 0.0)),
            "X-Finalize-Wait-Ms": self._format_ms_header(profile.get("finalize_wait_ms", 0.0)),
            "X-Synth-Ms": self._format_ms_header(profile.get("synth_ms", 0.0)),
            "X-Worker-Residual-Ms": self._format_ms_header(profile.get("worker_residual_ms", 0.0)),
            "X-Worker-Other-Ms": self._format_ms_header(profile.get("worker_other_ms", 0.0)),
            "X-Pack-Ms": self._format_ms_header(profile.get("pack_ms", 0.0)),
            "X-Worker-Total-Ms": self._format_ms_header(profile.get("worker_total_ms", 0.0)),
            "X-Api-Wait-Result-Ms": self._format_ms_header(profile.get("api_wait_result_ms", 0.0)),
            "X-Decode-Steps": str(int(profile.get("decode_steps", 0))),
            "X-Sample-Rate": str(int(sample_rate)),
            "X-Response-Overhead-Ms": self._format_ms_header(profile.get("response_overhead_ms", 0.0)),
            "X-Request-Other-Ms": self._format_ms_header(profile.get("request_other_ms", 0.0)),
            "X-Request-Total-Ms": self._format_ms_header(profile.get("request_total_ms", 0.0)),
        }
        headers.update(
            {
                "X-Prepare-Prompt-Text-Ms": self._format_ms_header(prepare_profile.get("prompt_text_features_ms", 0.0)),
                "X-Prepare-Target-Text-Ms": self._format_ms_header(prepare_profile.get("text_features_ms", 0.0)),
                "X-Prepare-Prompt-Text-CPU-Preprocess-Ms": self._format_ms_header(prepare_profile.get("prompt_text_cpu_preprocess_ms", 0.0)),
                "X-Prepare-Target-Text-CPU-Preprocess-Ms": self._format_ms_header(prepare_profile.get("text_cpu_preprocess_ms", 0.0)),
                "X-Prepare-Prompt-Text-CPU-Queue-Ms": self._format_ms_header(prepare_profile.get("prompt_text_cpu_queue_ms", 0.0)),
                "X-Prepare-Target-Text-CPU-Queue-Ms": self._format_ms_header(prepare_profile.get("text_cpu_queue_ms", 0.0)),
                "X-Prepare-Prompt-Text-Feature-Queue-Ms": self._format_ms_header(prepare_profile.get("prompt_text_feature_queue_ms", 0.0)),
                "X-Prepare-Target-Text-Feature-Queue-Ms": self._format_ms_header(prepare_profile.get("text_feature_queue_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_wait_ms", 0.0)),
                "X-Prepare-Target-Bert-Wait-Ms": self._format_ms_header(prepare_profile.get("text_bert_wait_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Admission-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_admission_wait_ms", 0.0)),
                "X-Prepare-Target-Bert-Admission-Wait-Ms": self._format_ms_header(prepare_profile.get("text_bert_admission_wait_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Queue-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_queue_wait_ms", 0.0)),
                "X-Prepare-Target-Bert-Queue-Wait-Ms": self._format_ms_header(prepare_profile.get("text_bert_queue_wait_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Batch-Collect-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_batch_collect_wait_ms", 0.0)),
                "X-Prepare-Target-Bert-Batch-Collect-Wait-Ms": self._format_ms_header(prepare_profile.get("text_bert_batch_collect_wait_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Forward-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_forward_ms", 0.0)),
                "X-Prepare-Target-Bert-Forward-Ms": self._format_ms_header(prepare_profile.get("text_bert_forward_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Pending-On-Enqueue-Peak": str(int(prepare_profile.get("prompt_text_bert_pending_depth_on_enqueue_peak", 0.0))),
                "X-Prepare-Target-Bert-Pending-On-Enqueue-Peak": str(int(prepare_profile.get("text_bert_pending_depth_on_enqueue_peak", 0.0))),
                "X-Prepare-Prompt-Bert-Pending-On-Collect-Peak": str(int(prepare_profile.get("prompt_text_bert_pending_depth_on_collect_peak", 0.0))),
                "X-Prepare-Target-Bert-Pending-On-Collect-Peak": str(int(prepare_profile.get("text_bert_pending_depth_on_collect_peak", 0.0))),
                "X-Prepare-Prompt-Bert-High-Pressure-Peak": str(int(prepare_profile.get("prompt_text_bert_high_pressure_mode_peak", 0.0))),
                "X-Prepare-Target-Bert-High-Pressure-Peak": str(int(prepare_profile.get("text_bert_high_pressure_mode_peak", 0.0))),
                "X-Prepare-Prompt-Bert-Batch-Window-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_batch_window_ms", 0.0)),
                "X-Prepare-Target-Bert-Batch-Window-Ms": self._format_ms_header(prepare_profile.get("text_bert_batch_window_ms", 0.0)),
                "X-Prepare-Text-Pair-Wall-Ms": self._format_ms_header(prepare_profile.get("text_feature_pair_ms", 0.0)),
                "X-Prepare-Text-CPU-Workers": str(int(prepare_profile.get("text_cpu_parallel_workers", 0.0))),
                "X-Prepare-Engine-GPU-Queue-Wait-Ms": self._format_ms_header(prepare_profile.get("engine_gpu_prepare_queue_wait_ms", 0.0)),
                "X-Prepare-Audio-Load-Ms": self._format_ms_header(prepare_profile.get("audio_load_ms", 0.0)),
                "X-Prepare-Audio-Stage-Wait-Ms": self._format_ms_header(prepare_profile.get("audio_stage_wait_ms", 0.0)),
                "X-Prepare-Prompt-Semantic-Ms": self._format_ms_header(prepare_profile.get("prompt_semantic_ms", 0.0)),
                "X-Prepare-Prompt-Semantic-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_semantic_wait_ms", 0.0)),
                "X-Prepare-Prompt-Semantic-CPU-Ms": self._format_ms_header(prepare_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)),
                "X-Prepare-Prompt-Semantic-Forward-Ms": self._format_ms_header(prepare_profile.get("prompt_semantic_forward_ms", 0.0)),
                "X-Prepare-Ref-Spec-Ms": self._format_ms_header(prepare_profile.get("ref_spec_ms", 0.0)),
                "X-Prepare-Ref-Spec-Wait-Ms": self._format_ms_header(prepare_profile.get("ref_spec_wait_ms", 0.0)),
                "X-Prepare-Ref-Bundle-Ms": self._format_ms_header(prepare_profile.get("ref_audio_bundle_ms", 0.0)),
                "X-Prepare-Tensorize-Ms": self._format_ms_header(prepare_profile.get("tensorize_ms", 0.0)),
                "X-Prepare-Inflight-On-Enter": str(int(prepare_profile.get("worker_prepare_inflight_on_enter", 0.0))),
                "X-Prepare-Inflight-Peak": str(int(prepare_profile.get("worker_prepare_peak_inflight", 0.0))),
            }
        )
        return headers

    def _build_scheduler_debug_request_profile(
        self,
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

    @staticmethod
    def _build_scheduler_debug_batch_profile(
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

    def _normalize_lang(self, value: str | None) -> str | None:
        if value in [None, ""]:
            return value
        return str(value).lower()

    @staticmethod
    def _aggregate_numeric_dicts(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for item in items:
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value)
        return totals

    def _apply_default_reference(self, req: dict) -> dict:
        normalized = dict(req)
        default_ref = self.reference_registry.get_default()
        if normalized.get("ref_audio_path") in [None, ""] and default_ref.ref_audio_path not in [None, ""]:
            normalized["ref_audio_path"] = default_ref.ref_audio_path
        if "text_lang" in normalized:
            normalized["text_lang"] = self._normalize_lang(normalized.get("text_lang"))
        if "prompt_lang" in normalized:
            normalized["prompt_lang"] = self._normalize_lang(normalized.get("prompt_lang"))
        return normalized

    def check_params(self, req: dict) -> Optional[str]:
        text = req.get("text", "")
        text_lang = req.get("text_lang", "")
        ref_audio_path = req.get("ref_audio_path", "")
        media_type = req.get("media_type", "wav")
        prompt_lang = req.get("prompt_lang", "")
        text_split_method = req.get("text_split_method", "cut5")

        if ref_audio_path in [None, ""]:
            return "ref_audio_path is required"
        if text in [None, ""]:
            return "text is required"
        if text_lang in [None, ""]:
            return "text_lang is required"
        if text_lang.lower() not in self.tts.configs.languages:
            return f"text_lang: {text_lang} is not supported in version {self.tts.configs.version}"
        if prompt_lang in [None, ""]:
            return "prompt_lang is required"
        if prompt_lang.lower() not in self.tts.configs.languages:
            return f"prompt_lang: {prompt_lang} is not supported in version {self.tts.configs.version}"
        if media_type not in ["wav", "raw", "ogg", "aac"]:
            return f"media_type: {media_type} is not supported"
        if text_split_method not in self.cut_method_names:
            return f"text_split_method:{text_split_method} is not supported"
        return None

    @staticmethod
    def _base_request_defaults() -> Dict[str, Any]:
        return {
            "request_id": None,
            "text": None,
            "text_lang": None,
            "ref_audio_path": None,
            "aux_ref_audio_paths": None,
            "prompt_text": "",
            "prompt_lang": None,
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "speed_factor": 1.0,
            "split_bucket": False,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "return_fragment": False,
            "fixed_length_chunk": False,
            "response_streaming": False,
            "parallel_infer": False,
            "repetition_penalty": 1.35,
            "sample_steps": 32,
            "super_sampling": False,
            "overlap_length": 2,
            "min_chunk_length": 16,
            "early_stop_num": -1,
            "ready_step": 0,
            "timeout_sec": None,
        }

    def _normalize_engine_request(
        self,
        payload: dict | NormalizedEngineRequest,
        *,
        request_id: str | None = None,
        normalize_streaming: bool = False,
        error_prefix: str = "request 参数非法: ",
    ) -> NormalizedEngineRequest:
        if isinstance(payload, NormalizedEngineRequest):
            normalized_payload = payload.to_payload()
        else:
            normalized_payload = self._base_request_defaults()
            normalized_payload.update(dict(payload))
        if request_id not in [None, ""]:
            normalized_payload["request_id"] = str(request_id)
        elif normalized_payload.get("request_id") in [None, ""]:
            raise ValueError("request_id is required after normalization")
        normalized_payload = self._apply_default_reference(normalized_payload)
        if normalize_streaming:
            normalized_payload = self._normalize_streaming_mode(normalized_payload)
        error = self.check_params(normalized_payload)
        if error is not None:
            raise ValueError(f"{error_prefix}{error}")
        timeout_sec = normalized_payload.get("timeout_sec")
        if timeout_sec in [None, ""]:
            parsed_timeout = None
        else:
            parsed_timeout = float(timeout_sec)
        aux_ref_audio_paths = normalized_payload.get("aux_ref_audio_paths")
        if aux_ref_audio_paths in [None, "", []]:
            normalized_aux_ref_audio_paths = None
        else:
            normalized_aux_ref_audio_paths = [str(item) for item in aux_ref_audio_paths]
        return NormalizedEngineRequest(
            request_id=str(normalized_payload["request_id"]),
            text=str(normalized_payload["text"]),
            text_lang=str(normalized_payload["text_lang"]),
            ref_audio_path=str(normalized_payload["ref_audio_path"]),
            prompt_lang=str(normalized_payload["prompt_lang"]),
            prompt_text="" if normalized_payload.get("prompt_text") is None else str(normalized_payload.get("prompt_text")),
            aux_ref_audio_paths=normalized_aux_ref_audio_paths,
            top_k=int(normalized_payload["top_k"]),
            top_p=float(normalized_payload["top_p"]),
            temperature=float(normalized_payload["temperature"]),
            repetition_penalty=float(normalized_payload["repetition_penalty"]),
            early_stop_num=int(normalized_payload.get("early_stop_num", -1)),
            ready_step=int(normalized_payload.get("ready_step", 0)),
            text_split_method=str(normalized_payload["text_split_method"]),
            batch_size=int(normalized_payload["batch_size"]),
            batch_threshold=float(normalized_payload["batch_threshold"]),
            split_bucket=bool(normalized_payload["split_bucket"]),
            speed_factor=float(normalized_payload["speed_factor"]),
            fragment_interval=float(normalized_payload["fragment_interval"]),
            seed=int(normalized_payload["seed"]),
            media_type=str(normalized_payload["media_type"]),
            streaming_mode=normalized_payload["streaming_mode"],
            return_fragment=bool(normalized_payload.get("return_fragment", False)),
            fixed_length_chunk=bool(normalized_payload.get("fixed_length_chunk", False)),
            response_streaming=bool(normalized_payload.get("response_streaming", False)),
            parallel_infer=bool(normalized_payload["parallel_infer"]),
            sample_steps=int(normalized_payload["sample_steps"]),
            super_sampling=bool(normalized_payload["super_sampling"]),
            overlap_length=int(normalized_payload["overlap_length"]),
            min_chunk_length=int(normalized_payload["min_chunk_length"]),
            timeout_sec=parsed_timeout,
        )

    @staticmethod
    def _normalize_streaming_mode(req: dict) -> dict:
        normalized = dict(req)
        streaming_mode = normalized.get("streaming_mode", False)
        return_fragment = normalized.get("return_fragment", False)
        if streaming_mode is False:
            normalized["streaming_mode"] = False
            normalized["return_fragment"] = False
            normalized["fixed_length_chunk"] = False
        elif streaming_mode == 0:
            normalized["streaming_mode"] = False
            normalized["return_fragment"] = False
            normalized["fixed_length_chunk"] = False
        elif streaming_mode == 1 or streaming_mode is True:
            normalized["streaming_mode"] = False
            normalized["return_fragment"] = True
            normalized["fixed_length_chunk"] = False
        elif streaming_mode == 2:
            normalized["streaming_mode"] = True
            normalized["return_fragment"] = False
            normalized["fixed_length_chunk"] = False
        elif streaming_mode == 3:
            normalized["streaming_mode"] = True
            normalized["return_fragment"] = False
            normalized["fixed_length_chunk"] = True
        else:
            raise ValueError("the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)")
        normalized["response_streaming"] = bool(normalized["streaming_mode"] or normalized["return_fragment"] or return_fragment)
        return normalized

    @staticmethod
    def _is_aux_ref_enabled(aux_ref_audio_paths: List[str] | None) -> bool:
        return aux_ref_audio_paths not in [None, [], ()]

    def _select_direct_backend(self, normalized: NormalizedEngineRequest) -> Tuple[str, str | None]:
        if normalized.response_streaming:
            if normalized.return_fragment or normalized.fixed_length_chunk:
                return "legacy_direct_fragment", "fragment_streaming_mode"
            return "legacy_direct_streaming", "streaming_mode"
        if self._is_aux_ref_enabled(normalized.aux_ref_audio_paths):
            return "legacy_direct_aux_ref", "aux_ref_audio_paths"
        if normalized.super_sampling:
            return "legacy_direct_super_sampling", "super_sampling"
        if normalized.prompt_text in [None, ""]:
            return "legacy_direct_missing_prompt", "missing_prompt_text"
        return "scheduler_v1_direct", None

    def _iter_legacy_direct_tts_bytes(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> Generator[bytes, None, None]:
        payload = normalized.to_payload()
        media_type = normalized.media_type
        request_id = normalized.request_id
        request_start = time.perf_counter()
        chunk_count = 0
        stream_total_bytes = 0
        first_chunk_ms: float | None = None
        self._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        try:
            with self.direct_tts_lock:
                tts_generator = self.tts.run(payload)
                first_chunk = True
                current_media_type = media_type
                for sr, chunk in tts_generator:
                    if first_chunk:
                        first_chunk_ms = max(0.0, (time.perf_counter() - request_start) * 1000.0)
                        self._update_request_state(
                            request_id,
                            EngineStatus.STREAMING,
                            {
                                "backend": backend,
                                "backend_mode": backend,
                                "fallback_reason": fallback_reason,
                                "sample_rate": int(sr),
                            },
                        )
                    if first_chunk and media_type == "wav":
                        header = wave_header_chunk(sample_rate=sr)
                        chunk_count += 1
                        stream_total_bytes += len(header)
                        yield header
                        current_media_type = "raw"
                        first_chunk = False
                    elif first_chunk:
                        first_chunk = False
                    packed_chunk = pack_audio(BytesIO(), chunk, sr, current_media_type).getvalue()
                    chunk_count += 1
                    stream_total_bytes += len(packed_chunk)
                    yield packed_chunk
        except Exception as exc:
            self._fail_request_state(request_id, str(exc))
            raise
        self._complete_request_state(
            request_id,
            dict(
                self._build_legacy_direct_profile(
                    backend=backend,
                    fallback_reason=fallback_reason,
                    request_start=request_start,
                    finished_at=time.perf_counter(),
                    audio_bytes=stream_total_bytes,
                    chunk_count=chunk_count,
                    stream_total_bytes=stream_total_bytes,
                    first_chunk_ms=first_chunk_ms,
                ),
                streaming_completed=True,
            ),
        )

    def _should_use_scheduler_backend_for_direct(self, req: dict | NormalizedEngineRequest) -> bool:
        if isinstance(req, NormalizedEngineRequest):
            normalized = req
        else:
            normalized = self._normalize_engine_request(
                req,
                request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
                normalize_streaming=True,
            )
        backend, _ = self._select_direct_backend(normalized)
        return backend == "scheduler_v1_direct"

    def _segment_direct_text(self, normalized: dict | NormalizedEngineRequest) -> List[str]:
        payload = normalized.to_payload() if isinstance(normalized, NormalizedEngineRequest) else normalized
        return self.tts.text_preprocessor.pre_seg_text(
            str(payload["text"]),
            str(payload["text_lang"]),
            str(payload.get("text_split_method", "cut5")),
        )

    def _build_segment_request(
        self,
        normalized: NormalizedEngineRequest,
        *,
        request_id: str,
        text: str,
    ) -> NormalizedEngineRequest:
        payload = normalized.to_payload()
        payload["request_id"] = request_id
        payload["text"] = text
        payload["streaming_mode"] = False
        payload["return_fragment"] = False
        payload["fixed_length_chunk"] = False
        payload["response_streaming"] = False
        return self._normalize_engine_request(payload, error_prefix="segment request 参数非法: ")

    async def _run_direct_tts_via_scheduler(self, normalized: NormalizedEngineRequest) -> DirectTTSExecution:
        request_start = time.perf_counter()
        request_id = normalized.request_id
        media_type = normalized.media_type
        segment_texts = self._segment_direct_text(normalized)
        if not segment_texts:
            raise ValueError("text preprocessing returned no valid segments")
        self._update_request_state(
            request_id,
            EngineStatus.CPU_PREPARING,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct", "segment_count": len(segment_texts)},
        )
        segment_specs: List[SchedulerRequestSpec] = []
        for segment_index, segment_text in enumerate(segment_texts):
            segment_request = self._build_segment_request(
                normalized,
                request_id=f"{request_id}_seg_{segment_index:03d}",
                text=segment_text,
            )
            segment_specs.append(self._build_scheduler_submit_spec(segment_request))

        prepared_items = await asyncio.gather(
            *[
                self._prepare_state_via_engine_gpu_queue(
                    spec=spec,
                    prepare_submit_at=time.perf_counter(),
                    engine_request_id=None,
                )
                for spec in segment_specs
            ]
        )
        prepare_profiles: List[Dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        done_futures: List[asyncio.Future] = []
        self._update_request_state(
            request_id,
            EngineStatus.READY_FOR_PREFILL,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct", "segment_count": len(segment_specs)},
        )
        for spec, (state, prepare_exec_started_at, prepare_exec_finished_at) in zip(segment_specs, prepared_items):
            prepare_wall_ms = max(0.0, (prepare_exec_finished_at - prepare_exec_started_at) * 1000.0)
            prepare_profile_total_ms = float(state.prepare_profile.get("wall_total_ms", prepare_wall_ms))
            prepare_profiles.append(
                {
                    "request_id": spec.request_id,
                    "prepare_wall_ms": prepare_wall_ms,
                    "prepare_profile_total_ms": prepare_profile_total_ms,
                    "prepare_profile": dict(state.prepare_profile),
                }
            )
            done_future = loop.create_future()
            done_futures.append(done_future)
            await self._enqueue_prepared_state_for_dispatch(
                state=state,
                speed_factor=float(normalized.speed_factor),
                sample_steps=int(normalized.sample_steps),
                media_type=media_type,
                prepare_wall_ms=prepare_wall_ms,
                prepare_profile_total_ms=prepare_profile_total_ms,
                done_loop=loop,
                done_future=done_future,
                engine_request_id=None,
                timeout_sec=normalized.timeout_sec,
            )
        self._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct"},
        )
        timeout_sec = float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0)
        jobs: List[SchedulerPendingJob] = list(await asyncio.wait_for(asyncio.gather(*done_futures), timeout=timeout_sec))
        for profile_item, job in zip(prepare_profiles, jobs):
            profile_item["engine_policy_wait_ms"] = float(job.engine_policy_wait_ms)
            profile_item["engine_dispatch_wait_ms"] = float(job.engine_dispatch_wait_ms)
        self._merge_request_state_profile(
            request_id,
            {
                "engine_policy_wait_ms": sum(float(job.engine_policy_wait_ms) for job in jobs),
                "engine_dispatch_wait_ms": sum(float(job.engine_dispatch_wait_ms) for job in jobs),
                "prepare_aggregate": self._aggregate_numeric_dicts(
                    [item["prepare_profile"] for item in prepare_profiles]
                ),
            },
        )

        sample_rate: int | None = None
        audio_parts: List[np.ndarray] = []
        worker_profiles: List[Dict[str, Any]] = []
        fragment_interval = float(normalized.fragment_interval)
        silence_chunk: Optional[np.ndarray] = None
        for job in jobs:
            if job.error is not None:
                raise RuntimeError(job.error)
            if job.audio_data is None or job.sample_rate is None or job.result is None:
                raise RuntimeError(f"{job.request_id} finished without audio result")
            if sample_rate is None:
                sample_rate = int(job.sample_rate)
                silence_samples = int(fragment_interval * float(sample_rate))
                if silence_samples > 0:
                    silence_chunk = np.zeros(silence_samples, dtype=np.int16)
            elif int(job.sample_rate) != sample_rate:
                raise RuntimeError("segment sample rate mismatch")
            audio_parts.append(job.audio_data)
            if silence_chunk is not None:
                audio_parts.append(silence_chunk.copy())
            worker_profiles.append(dict(job.result))
        if sample_rate is None or not audio_parts:
            raise RuntimeError("direct scheduler backend produced no audio")
        self._update_request_state(
            request_id,
            EngineStatus.FINALIZING,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct"},
        )
        merged_audio = np.concatenate(audio_parts, axis=0)
        pack_start = time.perf_counter()
        audio_bytes = pack_audio(BytesIO(), merged_audio, sample_rate, media_type).getvalue()
        pack_ms = max(0.0, (time.perf_counter() - pack_start) * 1000.0)
        direct_profile = self._build_direct_scheduler_profile(
            backend="scheduler_v1_direct",
            request_start=request_start,
            response_ready_at=time.perf_counter(),
            audio_bytes=len(audio_bytes),
            sample_rate=int(sample_rate),
            segment_texts=segment_texts,
            prepare_profiles=prepare_profiles,
            worker_profiles=worker_profiles,
            pack_ms=pack_ms,
            response_overhead_ms=0.0,
        )
        self._complete_request_state(
            request_id,
            dict(direct_profile, streaming_completed=False),
        )
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=audio_bytes,
            request_id=request_id,
        )

    def _run_legacy_direct_tts_blocking(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        normalized_payload = normalized.to_payload()
        request_id = normalized.request_id
        media_type = normalized.media_type
        request_start = time.perf_counter()
        self._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        with self.direct_tts_lock:
            tts_generator = self.tts.run(normalized_payload)
            try:
                sr, audio_data = next(tts_generator)
            except Exception as exc:
                self._fail_request_state(request_id, str(exc))
                raise
        self._update_request_state(
            request_id,
            EngineStatus.FINALIZING,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        pack_start = time.perf_counter()
        packed_audio = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
        pack_ms = max(0.0, (time.perf_counter() - pack_start) * 1000.0)
        self._complete_request_state(
            request_id,
            dict(
                self._build_legacy_direct_profile(
                    backend=backend,
                    fallback_reason=fallback_reason,
                    request_start=request_start,
                    finished_at=time.perf_counter(),
                    sample_rate=int(sr),
                    audio_bytes=len(packed_audio),
                    pack_ms=pack_ms,
                ),
                streaming_completed=False,
            ),
        )
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=packed_audio,
            request_id=request_id,
        )

    async def _run_direct_tts_via_legacy_backend(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        if normalized.response_streaming:
            return DirectTTSExecution(
                media_type=normalized.media_type,
                streaming=True,
                audio_generator=self._iter_legacy_direct_tts_bytes(
                    normalized,
                    backend=backend,
                    fallback_reason=fallback_reason,
                ),
                request_id=normalized.request_id,
            )
        return await asyncio.to_thread(
            self._run_legacy_direct_tts_blocking,
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    async def run_direct_tts_async(self, req: dict) -> DirectTTSExecution:
        normalized = self._normalize_engine_request(
            req,
            request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
            normalize_streaming=True,
            error_prefix="",
        )
        request_id = normalized.request_id
        media_type = normalized.media_type
        backend, fallback_reason = self._select_direct_backend(normalized)
        self._register_request_state(
            request_id=request_id,
            api_mode="tts",
            backend=backend,
            media_type=media_type,
            response_streaming=bool(normalized.response_streaming),
            deadline_ts=(
                time.perf_counter() + float(normalized.timeout_sec)
                if normalized.timeout_sec is not None
                else None
            ),
            meta=self._build_request_meta(normalized.to_payload()),
        )
        self._update_request_state(
            request_id,
            EngineStatus.VALIDATED,
            {
                "request_source": "direct_tts",
                "selected_backend": backend,
                "fallback_reason": fallback_reason,
            },
        )
        if backend == "scheduler_v1_direct":
            try:
                return await self._run_direct_tts_via_scheduler(normalized)
            except Exception as exc:
                self._fail_request_state(request_id, str(exc))
                raise
        return await self._run_direct_tts_via_legacy_backend(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    def run_direct_tts(self, req: dict) -> DirectTTSExecution:
        normalized = self._normalize_engine_request(
            req,
            request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
            normalize_streaming=True,
            error_prefix="",
        )
        request_id = normalized.request_id
        media_type = normalized.media_type
        backend, fallback_reason = self._select_direct_backend(normalized)
        if not self._has_active_request(request_id):
            self._register_request_state(
                request_id=request_id,
                api_mode="tts",
                backend=backend,
                media_type=media_type,
                response_streaming=bool(normalized.response_streaming),
                meta=self._build_request_meta(normalized.to_payload()),
            )
        self._update_request_state(
            request_id,
            EngineStatus.VALIDATED,
            {
                "request_source": "direct_tts",
                "selected_backend": backend,
                "fallback_reason": fallback_reason,
            },
        )
        if backend != "scheduler_v1_direct":
            if normalized.response_streaming:
                return DirectTTSExecution(
                    media_type=media_type,
                    streaming=True,
                    audio_generator=self._iter_legacy_direct_tts_bytes(
                        normalized,
                        backend=backend,
                        fallback_reason=fallback_reason,
                    ),
                    request_id=request_id,
                )
            return self._run_legacy_direct_tts_blocking(
                normalized,
                backend=backend,
                fallback_reason=fallback_reason,
            )
        if normalized.response_streaming:
            return DirectTTSExecution(
                media_type=media_type,
                streaming=True,
                audio_generator=self._iter_legacy_direct_tts_bytes(
                    normalized,
                    backend="legacy_direct_sync_compat",
                    fallback_reason="sync_direct_compat",
                ),
                request_id=request_id,
            )
        return self._run_legacy_direct_tts_blocking(
            normalized,
            backend="legacy_direct_sync_compat",
            fallback_reason="sync_direct_compat",
        )

    def _build_scheduler_request_specs(self, request_items: List[dict]) -> List[SchedulerRequestSpec]:
        specs: List[SchedulerRequestSpec] = []
        for index, payload in enumerate(request_items):
            normalized = self._normalize_engine_request(
                payload,
                request_id=str(payload.get("request_id") or f"req_{index:03d}"),
                error_prefix=f"request[{index}] 参数非法: ",
            )
            specs.append(normalized.to_scheduler_spec())
        return specs

    def _build_scheduler_submit_spec(self, payload: dict | NormalizedEngineRequest) -> SchedulerRequestSpec:
        normalized = self._normalize_engine_request(
            payload,
            request_id=(
                payload.request_id
                if isinstance(payload, NormalizedEngineRequest)
                else str(payload.get("request_id") or f"job_{uuid.uuid4().hex[:12]}")
            ),
        )
        return normalized.to_scheduler_spec()

    @staticmethod
    def _summarize_scheduler_states(states: List[T2SRequestState]) -> List[dict]:
        return [
            {
                "request_id": state.request_id,
                "ready_step": int(state.ready_step),
                "ref_audio_path": str(state.ref_audio_path),
                "prompt_semantic_len": int(state.prompt_semantic.shape[0]),
                "all_phone_len": int(state.all_phones.shape[0]),
                "bert_len": int(state.all_bert_features.shape[-1]),
                "norm_text": state.norm_text,
            }
            for state in states
        ]

    @staticmethod
    def _summarize_scheduler_finished(items: List[T2SFinishedItem]) -> List[dict]:
        return [
            {
                "request_id": item.request_id,
                "semantic_len": int(item.semantic_tokens.shape[0]),
                "finish_idx": int(item.finish_idx),
                "finish_reason": item.finish_reason,
            }
            for item in items
        ]

    async def run_scheduler_debug(self, request_items: List[dict], max_steps: int, seed: int) -> SchedulerDebugExecution:
        request_start = time.perf_counter()
        set_scheduler_seed(seed)
        specs = self._build_scheduler_request_specs(request_items)
        request_ids = [spec.request_id for spec in specs]
        for spec in specs:
            self._register_request_state(
                request_id=spec.request_id,
                api_mode="scheduler_debug",
                backend="scheduler_debug",
                media_type="wav",
                response_streaming=False,
                meta={
                    "text_len": len(spec.text),
                    "prompt_text_len": len(spec.prompt_text),
                    "text_lang": spec.text_lang,
                    "prompt_lang": spec.prompt_lang,
                    "ref_audio_path": str(spec.ref_audio_path),
                    "ready_step": int(spec.ready_step),
                },
            )
            self._update_request_state(spec.request_id, EngineStatus.VALIDATED, {"request_source": "scheduler_debug"})
            self._update_request_state(spec.request_id, EngineStatus.CPU_PREPARING, None)
        prepare_started_at = time.perf_counter()
        try:
            states = await self.scheduler_worker.prepare_states_batch_async(specs)
        except Exception as exc:
            for request_id in request_ids:
                self._fail_request_state(request_id, str(exc))
            raise
        prepare_finished_at = time.perf_counter()
        prepare_batch_wall_ms = max(0.0, (prepare_finished_at - prepare_started_at) * 1000.0)
        for state in states:
            self._update_request_state(
                state.request_id,
                EngineStatus.ACTIVE_DECODE,
                {
                    "prepare_profile": dict(state.prepare_profile),
                    "norm_text": state.norm_text,
                    "norm_prompt_text": state.norm_prompt_text,
                },
            )
        decode_started_at = time.perf_counter()
        try:
            finished = run_scheduler_continuous(self.tts.t2s_model.model, states, max_steps=int(max_steps))
        except Exception as exc:
            for request_id in request_ids:
                self._fail_request_state(request_id, str(exc))
            raise
        decode_finished_at = time.perf_counter()
        decode_batch_wall_ms = max(0.0, (decode_finished_at - decode_started_at) * 1000.0)
        request_total_ms = max(0.0, (decode_finished_at - request_start) * 1000.0)
        finished_map = {item.request_id: item for item in finished}
        request_profiles: List[Dict[str, Any]] = []
        for state in states:
            item = finished_map.get(state.request_id)
            if item is None:
                self._fail_request_state(state.request_id, "scheduler_debug finished without result")
                continue
            request_profile = self._build_scheduler_debug_request_profile(
                state=state,
                item=item,
                batch_request_count=len(states),
                prepare_batch_wall_ms=prepare_batch_wall_ms,
                decode_batch_wall_ms=decode_batch_wall_ms,
                batch_request_total_ms=request_total_ms,
            )
            request_profiles.append(
                {
                    "request_id": state.request_id,
                    "profile": dict(request_profile),
                }
            )
            self._complete_request_state(
                state.request_id,
                dict(request_profile),
            )
        return SchedulerDebugExecution(
            payload={
                "message": "success",
                "request_count": len(states),
                "max_steps": int(max_steps),
                "batch_profile": self._build_scheduler_debug_batch_profile(
                    request_count=len(states),
                    max_steps=int(max_steps),
                    prepare_batch_wall_ms=prepare_batch_wall_ms,
                    decode_batch_wall_ms=decode_batch_wall_ms,
                    request_total_ms=request_total_ms,
                    finished_items=finished,
                ),
                "requests": self._summarize_scheduler_states(states),
                "finished": self._summarize_scheduler_finished(finished),
                "request_profiles": request_profiles,
                "request_traces": self._collect_request_summaries(request_ids),
            }
        )

    async def run_scheduler_submit(self, payload: dict) -> SchedulerSubmitExecution:
        request_start = time.perf_counter()
        prepare_start = request_start
        normalized = self._normalize_engine_request(
            payload,
            request_id=str(payload.get("request_id") or f"job_{uuid.uuid4().hex[:12]}"),
        )
        spec = self._build_scheduler_submit_spec(normalized)
        deadline_ts = None
        timeout_sec = normalized.timeout_sec
        if timeout_sec is not None:
            try:
                deadline_ts = request_start + float(timeout_sec)
            except Exception:
                deadline_ts = None
        self._register_request_state(
            request_id=spec.request_id,
            api_mode="scheduler_submit",
            backend="scheduler_v1",
            media_type=normalized.media_type,
            response_streaming=False,
            deadline_ts=deadline_ts,
            meta=self._build_request_meta(normalized.to_payload()),
        )
        self._update_request_state(spec.request_id, EngineStatus.VALIDATED, {"request_source": "scheduler_submit"})
        spec_ready_at = time.perf_counter()
        prepare_spec_build_ms = max(0.0, (spec_ready_at - prepare_start) * 1000.0)
        self._update_request_state(spec.request_id, EngineStatus.CPU_PREPARING, {"prepare_spec_build_ms": prepare_spec_build_ms})
        try:
            state, prepare_exec_started_at, prepare_exec_finished_at = await self._prepare_state_via_engine_gpu_queue(
                spec=spec,
                prepare_submit_at=spec_ready_at,
                engine_request_id=spec.request_id,
            )
        except Exception as exc:
            self._fail_request_state(spec.request_id, str(exc))
            raise
        prepare_wall_ms = max(0.0, (prepare_exec_finished_at - spec_ready_at) * 1000.0)
        prepare_executor_queue_ms = max(0.0, (prepare_exec_started_at - spec_ready_at) * 1000.0)
        prepare_executor_run_ms = max(0.0, (prepare_exec_finished_at - prepare_exec_started_at) * 1000.0)
        prepare_profile = dict(state.prepare_profile)
        prepare_profile_total_ms = float(prepare_profile.get("wall_total_ms", prepare_wall_ms))
        prepare_profile_wall_ms = float(prepare_profile.get("wall_total_ms", prepare_wall_ms))
        prepare_other_ms = max(0.0, prepare_wall_ms - prepare_spec_build_ms - prepare_executor_queue_ms - prepare_executor_run_ms)
        self._update_request_state(
            spec.request_id,
            EngineStatus.READY_FOR_PREFILL,
            {
                "prepare_wall_ms": prepare_wall_ms,
                "prepare_profile_total_ms": prepare_profile_total_ms,
                "prepare_profile": prepare_profile,
            },
        )
        api_after_prepare_start = time.perf_counter()
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        await self._enqueue_prepared_state_for_dispatch(
            state=state,
            speed_factor=float(normalized.speed_factor),
            sample_steps=int(normalized.sample_steps),
            media_type=normalized.media_type,
            prepare_wall_ms=prepare_wall_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            done_loop=loop,
            done_future=done_future,
            engine_request_id=spec.request_id,
            timeout_sec=normalized.timeout_sec,
        )
        api_after_prepare_ms = max(0.0, (time.perf_counter() - api_after_prepare_start) * 1000.0)
        try:
            job = await asyncio.wait_for(done_future, timeout=float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0))
        except Exception as exc:
            self._fail_request_state(spec.request_id, str(exc))
            raise
        wait_return_at = time.perf_counter()
        if job.error is not None:
            raise RuntimeError(job.error)
        if job.audio_data is None or job.sample_rate is None or job.result is None:
            self._fail_request_state(spec.request_id, f"{job.request_id} finished without audio result")
            raise RuntimeError(f"{job.request_id} finished without audio result")
        pack_start = time.perf_counter()
        audio_data = pack_audio(BytesIO(), job.audio_data, int(job.sample_rate), job.media_type).getvalue()
        pack_end = time.perf_counter()
        pack_ms = (pack_end - pack_start) * 1000.0
        api_wait_result_ms = 0.0
        if job.result_ready_time is not None:
            api_wait_result_ms = max(0.0, (wait_return_at - job.result_ready_time) * 1000.0)
        response_ready_at = time.perf_counter()
        response_overhead_ms = max(0.0, (response_ready_at - pack_end) * 1000.0)
        submit_profile = self._build_scheduler_submit_profile(
            backend="scheduler_v1",
            request_start=request_start,
            response_ready_at=response_ready_at,
            audio_bytes=len(audio_data),
            sample_rate=int(job.sample_rate),
            prepare_spec_build_ms=prepare_spec_build_ms,
            prepare_wall_ms=prepare_wall_ms,
            prepare_executor_queue_ms=prepare_executor_queue_ms,
            prepare_executor_run_ms=prepare_executor_run_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            prepare_profile_wall_ms=prepare_profile_wall_ms,
            prepare_other_ms=prepare_other_ms,
            engine_policy_wait_ms=float(job.result.get("engine_policy_wait_ms", 0.0)),
            api_after_prepare_ms=api_after_prepare_ms,
            api_wait_result_ms=api_wait_result_ms,
            pack_ms=pack_ms,
            response_overhead_ms=response_overhead_ms,
            worker_profile=dict(job.result or {}),
        )
        headers = self._build_scheduler_submit_headers(
            request_id=job.request_id,
            media_type=job.media_type,
            sample_rate=int(job.sample_rate),
            profile=submit_profile,
        )
        self._merge_request_state_profile(
            spec.request_id,
            dict(submit_profile, response_headers_emitted=True),
        )
        return SchedulerSubmitExecution(audio_bytes=audio_data, media_type=f"audio/{job.media_type}", headers=headers)
