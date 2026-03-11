from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SActiveBatch, T2SFinishedItem, T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_bridge import EngineBridgeFacade
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineDispatchTask, EngineRequestState, SchedulerFinalizeTask, SchedulerPendingJob


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
        super_sampling: bool,
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
            super_sampling=super_sampling,
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
