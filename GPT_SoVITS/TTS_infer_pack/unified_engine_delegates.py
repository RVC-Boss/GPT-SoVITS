from __future__ import annotations

import asyncio
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SActiveBatch, T2SFinishedItem, T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_api import EngineApiFacade
from GPT_SoVITS.TTS_infer_pack.unified_engine_bridge import EngineBridgeFacade
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import DirectTTSExecution, EngineDispatchTask, EngineRequestState, NormalizedEngineRequest, SchedulerDebugExecution, SchedulerFinalizeTask, SchedulerPendingJob, SchedulerSubmitExecution
from GPT_SoVITS.TTS_infer_pack.unified_engine_runtime import EngineRuntimeFacade


class EngineBridgeDelegates:
    def _register_request_state(
        self,
        request_id: str,
        api_mode: str,
        backend: str,
        media_type: str,
        response_streaming: bool,
        deadline_ts: float | None = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> EngineRequestState:
        return self.bridge_facade._register_request_state(
            request_id=request_id,
            api_mode=api_mode,
            backend=backend,
            media_type=media_type,
            response_streaming=response_streaming,
            deadline_ts=deadline_ts,
            meta=meta,
        )

    def _update_request_state(self, request_id: str, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.bridge_facade._update_request_state(request_id, status, extra)

    def _merge_request_state_profile(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.bridge_facade._merge_request_state_profile(request_id, extra)

    def _snapshot_engine_prepare_state(self) -> Dict[str, Any]:
        return self.bridge_facade._snapshot_engine_prepare_state()

    def _snapshot_engine_finalize_state(self) -> Dict[str, Any]:
        return self.bridge_facade._snapshot_engine_finalize_state()

    def _snapshot_engine_dispatch_state(self) -> Dict[str, Any]:
        return self.bridge_facade._snapshot_engine_dispatch_state()

    def _register_engine_job(self, job: SchedulerPendingJob) -> None:
        self.bridge_facade._register_engine_job(job)

    def _get_engine_job(self, request_id: str) -> SchedulerPendingJob | None:
        return self.bridge_facade._get_engine_job(request_id)

    def _pop_engine_job(self, request_id: str) -> SchedulerPendingJob | None:
        return self.bridge_facade._pop_engine_job(request_id)

    def _snapshot_engine_job_registry(self) -> Dict[str, Any]:
        return self.bridge_facade._snapshot_engine_job_registry()

    def _is_engine_drained(self) -> bool:
        return self.bridge_facade._is_engine_drained()

    def _record_engine_job_done(self, request_id: str) -> None:
        self.bridge_facade._record_engine_job_done(request_id)

    def _complete_engine_job(
        self,
        job: SchedulerPendingJob,
        item: T2SFinishedItem,
        *,
        sample_rate: int,
        audio_data: np.ndarray,
    ) -> None:
        self.bridge_facade._complete_engine_job(job, item, sample_rate=sample_rate, audio_data=audio_data)

    def _fail_engine_jobs(self, request_ids: List[str], error: str) -> None:
        self.bridge_facade._fail_engine_jobs(request_ids, error)

    def _add_engine_prefill_time(self, jobs: List[SchedulerPendingJob], elapsed_s: float) -> None:
        self.bridge_facade._add_engine_prefill_time(jobs, elapsed_s)

    def _add_engine_merge_time(self, request_ids: List[str], elapsed_s: float) -> None:
        self.bridge_facade._add_engine_merge_time(request_ids, elapsed_s)

    def _add_engine_decode_time(self, request_ids: List[str], elapsed_s: float) -> None:
        self.bridge_facade._add_engine_decode_time(request_ids, elapsed_s)

    def _enqueue_engine_finished_items(self, items: List[T2SFinishedItem]) -> None:
        self.bridge_facade._enqueue_engine_finished_items(items)

    def _snapshot_engine_decode_pending_queue_state(self) -> Dict[str, Any]:
        return self.bridge_facade._snapshot_engine_decode_pending_queue_state()

    @staticmethod
    def _summarize_active_batch(active_batch: T2SActiveBatch | None) -> Dict[str, Any]:
        return EngineBridgeFacade._summarize_active_batch(active_batch)

    def _refresh_engine_decode_runtime_state(self, last_event: str) -> None:
        self.bridge_facade._refresh_engine_decode_runtime_state(last_event)

    def _update_engine_decode_runtime_state(self, snapshot: Dict[str, Any]) -> None:
        self.bridge_facade._update_engine_decode_runtime_state(snapshot)

    def _snapshot_engine_decode_runtime_state(self) -> Dict[str, Any]:
        return self.bridge_facade._snapshot_engine_decode_runtime_state()

    def _snapshot_engine_arbiter_state(self) -> Dict[str, Any]:
        return self.bridge_facade._snapshot_engine_arbiter_state()

    def _notify_engine_arbiter(self) -> None:
        self.bridge_facade._notify_engine_arbiter()

    def _enqueue_engine_decode_pending_job(self, job: SchedulerPendingJob) -> None:
        self.bridge_facade._enqueue_engine_decode_pending_job(job)

    def _take_engine_decode_pending_jobs_nonblocking(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        return self.bridge_facade._take_engine_decode_pending_jobs_nonblocking(wait_for_batch)

    def _peek_queue_age_ms(self, queue_name: str) -> float:
        return self.bridge_facade._peek_queue_age_ms(queue_name)

    def _engine_has_pending_work(self) -> bool:
        return self.bridge_facade._engine_has_pending_work()

    async def _prepare_state_via_engine_gpu_queue(
        self,
        *,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
        engine_request_id: str | None,
    ) -> tuple[T2SRequestState, float, float]:
        return await self.bridge_facade._prepare_state_via_engine_gpu_queue(
            spec=spec,
            prepare_submit_at=prepare_submit_at,
            engine_request_id=engine_request_id,
        )

    def _enqueue_worker_finished_for_finalize(self, tasks: List[SchedulerFinalizeTask]) -> None:
        self.bridge_facade._enqueue_worker_finished_for_finalize(tasks)

    def _take_engine_finalize_batch_nonblocking(self) -> List[SchedulerFinalizeTask]:
        return self.bridge_facade._take_engine_finalize_batch_nonblocking()

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
    ) -> EngineDispatchTask:
        return await self.bridge_facade._enqueue_prepared_state_for_dispatch(
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

    def _mark_arbiter_tick(self, *, stage: str, reason: str, policy_allowed: bool) -> None:
        self.bridge_facade._mark_arbiter_tick(stage=stage, reason=reason, policy_allowed=policy_allowed)

    def _select_engine_stage(self) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        return self.bridge_facade._select_engine_stage()

    def _run_engine_prepare_once(self) -> bool:
        return self.bridge_facade._run_engine_prepare_once()

    def _run_engine_finalize_once(self) -> bool:
        return self.bridge_facade._run_engine_finalize_once()

    def _run_engine_dispatch_once(self, policy_snapshot: Dict[str, Any], worker_state: Dict[str, Any]) -> bool:
        return self.bridge_facade._run_engine_dispatch_once(policy_snapshot, worker_state)

    def _run_engine_decode_runtime_once(self) -> bool:
        return self.bridge_facade._run_engine_decode_runtime_once()

    def _run_engine_arbiter_loop(self) -> None:
        self.bridge_facade._run_engine_arbiter_loop()

    def _complete_request_state(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.bridge_facade._complete_request_state(request_id, extra)

    def _fail_request_state(self, request_id: str, error: str) -> None:
        self.bridge_facade._fail_request_state(request_id, error)

    def _snapshot_request_registry(self) -> Dict[str, Any]:
        return self.bridge_facade._snapshot_request_registry()


class EngineApiDelegates:
    def _collect_request_summaries(self, request_ids: Sequence[str]) -> List[Dict[str, Any]]:
        return self.api_facade._collect_request_summaries(request_ids)

    def _has_active_request(self, request_id: str) -> bool:
        return self.api_facade._has_active_request(request_id)

    @staticmethod
    def _build_request_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
        return EngineApiFacade._build_request_meta(payload)

    @staticmethod
    def _sum_profile_field(items: Sequence[Dict[str, Any]], key: str) -> float:
        return EngineApiFacade._sum_profile_field(items, key)

    def _build_direct_segment_trace(
        self,
        segment_texts: Sequence[str],
        prepare_profiles: Sequence[Dict[str, Any]],
        worker_profiles: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self.api_facade._build_direct_segment_trace(segment_texts, prepare_profiles, worker_profiles)

    def _build_direct_scheduler_profile(self, **kwargs: Any) -> Dict[str, Any]:
        return self.api_facade._build_direct_scheduler_profile(**kwargs)

    def _build_legacy_direct_profile(self, **kwargs: Any) -> Dict[str, Any]:
        return self.api_facade._build_legacy_direct_profile(**kwargs)

    def _build_scheduler_submit_profile(self, **kwargs: Any) -> Dict[str, Any]:
        return self.api_facade._build_scheduler_submit_profile(**kwargs)

    @staticmethod
    def _format_ms_header(value: Any) -> str:
        return EngineApiFacade._format_ms_header(value)

    def _build_scheduler_submit_headers(
        self,
        *,
        request_id: str,
        media_type: str,
        sample_rate: int,
        profile: Dict[str, Any],
    ) -> Dict[str, str]:
        return self.api_facade._build_scheduler_submit_headers(
            request_id=request_id,
            media_type=media_type,
            sample_rate=sample_rate,
            profile=profile,
        )

    def _build_scheduler_debug_request_profile(self, **kwargs: Any) -> Dict[str, Any]:
        return self.api_facade._build_scheduler_debug_request_profile(**kwargs)

    @staticmethod
    def _build_scheduler_debug_batch_profile(**kwargs: Any) -> Dict[str, Any]:
        return EngineApiFacade._build_scheduler_debug_batch_profile(**kwargs)

    def _normalize_lang(self, value: str | None) -> str | None:
        return self.api_facade._normalize_lang(value)

    @staticmethod
    def _aggregate_numeric_dicts(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        return EngineApiFacade._aggregate_numeric_dicts(items)

    def _apply_default_reference(self, req: dict) -> dict:
        return self.api_facade._apply_default_reference(req)

    def check_params(self, req: dict) -> Optional[str]:
        return self.api_facade.check_params(req)

    @staticmethod
    def _base_request_defaults() -> Dict[str, Any]:
        return EngineApiFacade._base_request_defaults()

    def _normalize_engine_request(
        self,
        payload: dict | NormalizedEngineRequest,
        *,
        request_id: str | None = None,
        normalize_streaming: bool = False,
        error_prefix: str = "request 参数非法: ",
    ) -> NormalizedEngineRequest:
        return self.api_facade._normalize_engine_request(
            payload,
            request_id=request_id,
            normalize_streaming=normalize_streaming,
            error_prefix=error_prefix,
        )

    @staticmethod
    def _normalize_streaming_mode(req: dict) -> dict:
        return EngineApiFacade._normalize_streaming_mode(req)

    @staticmethod
    def _is_aux_ref_enabled(aux_ref_audio_paths: List[str] | None) -> bool:
        return EngineApiFacade._is_aux_ref_enabled(aux_ref_audio_paths)

    def _select_direct_backend(self, normalized: NormalizedEngineRequest) -> Tuple[str, str | None]:
        return self.api_facade._select_direct_backend(normalized)

    def _iter_legacy_direct_tts_bytes(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> Generator[bytes, None, None]:
        return self.api_facade._iter_legacy_direct_tts_bytes(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    def _should_use_scheduler_backend_for_direct(self, req: dict | NormalizedEngineRequest) -> bool:
        return self.api_facade._should_use_scheduler_backend_for_direct(req)

    def _segment_direct_text(self, normalized: dict | NormalizedEngineRequest) -> List[str]:
        return self.api_facade._segment_direct_text(normalized)

    def _build_segment_request(
        self,
        normalized: NormalizedEngineRequest,
        *,
        request_id: str,
        text: str,
    ) -> NormalizedEngineRequest:
        return self.api_facade._build_segment_request(normalized, request_id=request_id, text=text)

    async def _run_direct_tts_via_scheduler(self, normalized: NormalizedEngineRequest) -> DirectTTSExecution:
        return await self.api_facade._run_direct_tts_via_scheduler(normalized)

    def _run_legacy_direct_tts_blocking(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        return self.api_facade._run_legacy_direct_tts_blocking(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    async def _run_direct_tts_via_legacy_backend(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        return await self.api_facade._run_direct_tts_via_legacy_backend(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

class EngineRuntimeDelegates:
    @staticmethod
    def _safe_component_snapshot(component: Any) -> Dict[str, Any] | None:
        return EngineRuntimeFacade._safe_component_snapshot(component)

    def _build_stage_counters(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.runtime_facade._build_stage_counters(request_registry, worker_state)

    def _build_engine_policy_snapshot(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.runtime_facade._build_engine_policy_snapshot(request_registry, worker_state)

    async def _wait_for_engine_policy_admission(
        self,
        *,
        request_id: str | None,
        timeout_sec: float | None,
    ) -> tuple[float, Dict[str, Any]]:
        return await self.engine_policy_arbiter.wait_for_policy_admission(
            request_id=request_id,
            timeout_sec=timeout_sec,
        )

    def _build_stage_summary(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.runtime_facade._build_stage_summary(request_registry, worker_state)

    def _wait_for_safe_reload(self, timeout_sec: float = 300.0) -> None:
        self.runtime_facade._wait_for_safe_reload(timeout_sec=timeout_sec)
