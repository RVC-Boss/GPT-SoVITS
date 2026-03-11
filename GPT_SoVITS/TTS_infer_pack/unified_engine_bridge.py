from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import numpy as np

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SActiveBatch, T2SFinishedItem, T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineDecodeRuntimeOwner, EngineDispatchTask, EngineRequestState, EngineStatus, SchedulerFinalizeTask, SchedulerPendingJob


class EngineBridgeFacade:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    @property
    def request_registry(self):
        return self.owner.request_registry

    @property
    def engine_prepare_queue_owner(self):
        return self.owner.engine_prepare_queue_owner

    @property
    def engine_finalize_queue_owner(self):
        return self.owner.engine_finalize_queue_owner

    @property
    def engine_dispatch_queue_owner(self):
        return self.owner.engine_dispatch_queue_owner

    @property
    def engine_decode_runtime_owner(self):
        return self.owner.engine_decode_runtime_owner

    @property
    def engine_job_registry(self):
        return self.owner.engine_job_registry

    @property
    def scheduler_worker(self):
        return self.owner.scheduler_worker

    @property
    def engine_stage_coordinator(self):
        return self.owner.engine_stage_coordinator

    @property
    def engine_policy_arbiter(self):
        return self.owner.engine_policy_arbiter

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
        return self.request_registry.register(
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
        self.request_registry.update(request_id, status, extra)

    def _merge_request_state_profile(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.request_registry.merge_profile(request_id, extra)

    def _complete_request_state(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.request_registry.complete(request_id, extra)

    def _fail_request_state(self, request_id: str, error: str) -> None:
        self.request_registry.fail(request_id, error)

    def _snapshot_request_registry(self) -> Dict[str, Any]:
        return self.request_registry.snapshot()

    def _snapshot_engine_prepare_state(self) -> Dict[str, Any]:
        return self.engine_prepare_queue_owner.snapshot(max_request_ids=16)

    def _snapshot_engine_finalize_state(self) -> Dict[str, Any]:
        return self.engine_finalize_queue_owner.snapshot(max_request_ids=16)

    def _snapshot_engine_dispatch_state(self) -> Dict[str, Any]:
        return self.engine_dispatch_queue_owner.snapshot(
            max_request_ids=16,
            extra={"last_policy_snapshot": dict(self.owner.engine_dispatch_last_snapshot or {})},
        )

    def _register_engine_job(self, job: SchedulerPendingJob) -> None:
        self.engine_job_registry.register(job, keep_job=True)

    def _get_engine_job(self, request_id: str) -> SchedulerPendingJob | None:
        return self.engine_job_registry.get(request_id)

    def _pop_engine_job(self, request_id: str) -> SchedulerPendingJob | None:
        return self.engine_job_registry.pop(request_id)

    def _snapshot_engine_job_registry(self) -> Dict[str, Any]:
        return self.engine_job_registry.snapshot(max_request_ids=32)

    def _is_engine_drained(self) -> bool:
        prepare_empty = self.engine_prepare_queue_owner.is_drained()
        dispatch_empty = self.engine_dispatch_queue_owner.is_drained()
        finalize_empty = self.engine_finalize_queue_owner.is_drained()
        decode_pending_empty = not self.engine_decode_runtime_owner.has_pending_jobs()
        job_empty = self.engine_job_registry.is_empty()
        worker_state = self.scheduler_worker.snapshot()
        return bool(
            prepare_empty
            and dispatch_empty
            and finalize_empty
            and decode_pending_empty
            and job_empty
            and self.engine_decode_runtime_owner.get_active_batch() is None
            and int(worker_state.get("prepare_inflight", 0)) <= 0
            and int(worker_state.get("finalize_inflight", 0)) <= 0
            and int(worker_state.get("finalize_pending", 0)) <= 0
        )

    def _record_engine_job_done(self, request_id: str) -> None:
        self.engine_job_registry.mark_finished_and_remove(request_id)
        self.scheduler_worker.record_external_job_done(request_id)

    def _complete_engine_job(
        self,
        job: SchedulerPendingJob,
        item: T2SFinishedItem,
        *,
        sample_rate: int,
        audio_data: np.ndarray,
    ) -> None:
        completion_bridge = self.scheduler_worker.completion_bridge
        completion_bridge.build_completed_job_result(job, item, sample_rate=sample_rate, audio_data=audio_data)
        completion_bridge.complete_job(
            job,
            runtime_request_id=job.engine_request_id,
            runtime_extra=completion_bridge.build_runtime_complete_payload(job, item, sample_rate=sample_rate),
            on_job_finished=lambda rid=item.request_id: self._record_engine_job_done(rid),
        )

    def _fail_engine_jobs(self, request_ids: List[str], error: str) -> None:
        if not request_ids:
            return
        completion_bridge = self.scheduler_worker.completion_bridge
        for request_id in request_ids:
            job = self._get_engine_job(request_id)
            if job is None:
                continue
            completion_bridge.fail_job(
                job,
                error=error,
                on_job_finished=lambda rid=request_id: self._record_engine_job_done(rid),
            )

    def _add_engine_prefill_time(self, jobs: List[SchedulerPendingJob], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        for job in jobs:
            job.prefill_ms += delta_ms

    def _add_engine_merge_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        for request_id in request_ids:
            job = self._get_engine_job(request_id)
            if job is not None:
                job.merge_ms += delta_ms

    def _add_engine_decode_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        activate_request_ids: List[str] = []
        for request_id in request_ids:
            job = self._get_engine_job(request_id)
            if job is None:
                continue
            if job.decode_steps == 0:
                activate_request_ids.append(job.engine_request_id)
            job.decode_ms += delta_ms
            job.decode_steps += 1
        for engine_request_id in activate_request_ids:
            self._update_request_state(engine_request_id, EngineStatus.ACTIVE_DECODE, None)

    def _enqueue_engine_finished_items(self, items: List[T2SFinishedItem]) -> None:
        if not items:
            return
        enqueued_at = time.perf_counter()
        tasks = [SchedulerFinalizeTask(request_id=item.request_id, item=item, enqueued_time=enqueued_at) for item in items]
        self._enqueue_worker_finished_for_finalize(tasks)

    def _snapshot_engine_decode_pending_queue_state(self) -> Dict[str, Any]:
        return self.engine_decode_runtime_owner.snapshot_pending_queue_state()

    @staticmethod
    def _summarize_active_batch(active_batch: T2SActiveBatch | None) -> Dict[str, Any]:
        return EngineDecodeRuntimeOwner.summarize_active_batch(active_batch)

    def _refresh_engine_decode_runtime_state(self, last_event: str) -> None:
        self.engine_decode_runtime_owner.refresh_state(last_event)

    def _update_engine_decode_runtime_state(self, snapshot: Dict[str, Any]) -> None:
        if not snapshot:
            return
        if self.scheduler_worker.is_engine_decode_control_enabled():
            return
        self.engine_decode_runtime_owner.update_from_worker_snapshot(snapshot)

    def _snapshot_engine_decode_runtime_state(self) -> Dict[str, Any]:
        return self.engine_decode_runtime_owner.snapshot_state()

    def _snapshot_engine_arbiter_state(self) -> Dict[str, Any]:
        return self.engine_policy_arbiter.snapshot_state()

    def _notify_engine_arbiter(self) -> None:
        self.engine_policy_arbiter.notify()

    def _enqueue_engine_decode_pending_job(self, job: SchedulerPendingJob) -> None:
        self.engine_stage_coordinator.decode_runtime_owner.enqueue_pending_job(job)
        self._notify_engine_arbiter()

    def _take_engine_decode_pending_jobs_nonblocking(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        return self.engine_stage_coordinator.decode_runtime_owner.take_pending_jobs_nonblocking(wait_for_batch)

    def _peek_queue_age_ms(self, queue_name: str) -> float:
        return self.engine_stage_coordinator.peek_queue_age_ms(queue_name)

    def _engine_has_pending_work(self) -> bool:
        return self.engine_stage_coordinator.has_pending_work()

    async def _prepare_state_via_engine_gpu_queue(
        self,
        *,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
        engine_request_id: str | None,
    ) -> tuple[T2SRequestState, float, float]:
        return await self.engine_stage_coordinator.prepare_state_via_engine_gpu_queue(
            spec=spec,
            prepare_submit_at=prepare_submit_at,
            engine_request_id=engine_request_id,
        )

    def _enqueue_worker_finished_for_finalize(self, tasks: List[SchedulerFinalizeTask]) -> None:
        self.engine_stage_coordinator.enqueue_worker_finished_for_finalize(tasks)

    def _take_engine_finalize_batch_nonblocking(self) -> List[SchedulerFinalizeTask]:
        return self.engine_stage_coordinator.take_engine_finalize_batch_nonblocking()

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
        return await self.engine_stage_coordinator.enqueue_prepared_state_for_dispatch(
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
        self.engine_policy_arbiter.mark_tick(stage=stage, reason=reason, policy_allowed=policy_allowed)

    def _select_engine_stage(self) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        stage, reason, policy_snapshot, worker_state = self.engine_policy_arbiter.select_stage()
        self.owner.engine_dispatch_last_snapshot = dict(policy_snapshot)
        return stage, reason, policy_snapshot, worker_state

    def _run_engine_prepare_once(self) -> bool:
        return self.engine_stage_coordinator.run_engine_prepare_once()

    def _run_engine_finalize_once(self) -> bool:
        return self.engine_stage_coordinator.run_engine_finalize_once()

    def _run_engine_dispatch_once(self, policy_snapshot: Dict[str, Any], worker_state: Dict[str, Any]) -> bool:
        return self.engine_stage_coordinator.run_engine_dispatch_once(policy_snapshot, worker_state)

    def _run_engine_decode_runtime_once(self) -> bool:
        return self.engine_stage_coordinator.run_engine_decode_runtime_once()

    def _run_engine_arbiter_loop(self) -> None:
        self.engine_stage_coordinator.run_engine_arbiter_loop()
