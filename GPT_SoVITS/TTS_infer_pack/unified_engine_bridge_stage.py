from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineDispatchTask, SchedulerFinalizeTask, SchedulerPendingJob


class EngineStageBridgeFacade:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    @property
    def engine_decode_runtime_owner(self):
        return self.owner.engine_decode_runtime_owner

    @property
    def scheduler_worker(self):
        return self.owner.scheduler_worker

    @property
    def engine_stage_coordinator(self):
        return self.owner.engine_stage_coordinator

    def _snapshot_engine_decode_pending_queue_state(self) -> Dict[str, Any]:
        return self.engine_decode_runtime_owner.snapshot_pending_queue_state()

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

    def _enqueue_engine_decode_pending_job(self, job: SchedulerPendingJob) -> None:
        self.engine_decode_runtime_owner.enqueue_pending_job(job)
        self.owner.engine_policy_arbiter.notify()

    def _take_engine_decode_pending_jobs_nonblocking(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        return self.engine_decode_runtime_owner.take_pending_jobs_nonblocking(wait_for_batch)

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

    def _run_engine_prepare_once(self) -> bool:
        return self.engine_stage_coordinator.run_engine_prepare_once()

    def _run_engine_finalize_once(self) -> bool:
        return self.engine_stage_coordinator.run_engine_finalize_once()

    def _run_engine_dispatch_once(self, policy_snapshot: Dict[str, Any], worker_state: Dict[str, Any]) -> bool:
        return self.engine_stage_coordinator.run_engine_dispatch_once(policy_snapshot, worker_state)

    def _run_engine_decode_runtime_once(self) -> bool:
        return self.engine_stage_coordinator.run_engine_decode_runtime_once()

    def _run_engine_arbiter_loop(self) -> None:
        return self.engine_stage_coordinator.run_engine_arbiter_loop()
