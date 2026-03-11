from __future__ import annotations

import asyncio
import time
from typing import Dict

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineDispatchTask


class EngineDispatchStageMixin:
    async def enqueue_prepared_state_for_dispatch(
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
        task = EngineDispatchTask(
            request_id=state.request_id,
            state=state,
            speed_factor=float(speed_factor),
            sample_steps=int(sample_steps),
            media_type=media_type,
            super_sampling=bool(super_sampling),
            prepare_wall_ms=float(prepare_wall_ms),
            prepare_profile_total_ms=float(prepare_profile_total_ms),
            done_loop=done_loop,
            done_future=done_future,
            engine_request_id=engine_request_id or state.request_id,
            timeout_sec=timeout_sec,
            enqueue_time=time.perf_counter(),
        )
        self.dispatch_queue_owner.enqueue(task)
        self.notify_arbiter()
        self.merge_request_state_profile(
            task.engine_request_id or task.request_id,
            {
                "engine_dispatch_queue_depth_on_enqueue": int(
                    self.snapshot_engine_dispatch_state()["waiting_count"]
                ),
            },
        )
        return task

    def run_engine_dispatch_once(self, policy_snapshot: Dict[str, object], worker_state: Dict[str, object]) -> bool:
        if not bool(policy_snapshot.get("allowed", True)):
            return False
        dispatch_task = self.dispatch_queue_owner.pop_left()
        if dispatch_task is None:
            return False
        dispatched_at = time.perf_counter()
        dispatch_wait_ms = max(0.0, (dispatched_at - dispatch_task.enqueue_time) * 1000.0)
        dispatch_task.engine_policy_wait_ms = float(dispatch_wait_ms)
        dispatch_task.engine_dispatch_wait_ms = float(dispatch_wait_ms)
        dispatch_task.engine_policy_snapshot = dict(policy_snapshot)
        try:
            worker_job = self.scheduler_worker.submit(
                state=dispatch_task.state,
                speed_factor=dispatch_task.speed_factor,
                sample_steps=dispatch_task.sample_steps,
                media_type=dispatch_task.media_type,
                super_sampling=dispatch_task.super_sampling,
                prepare_wall_ms=dispatch_task.prepare_wall_ms,
                prepare_profile_total_ms=dispatch_task.prepare_profile_total_ms,
                done_loop=dispatch_task.done_loop,
                done_future=dispatch_task.done_future,
                engine_request_id=dispatch_task.engine_request_id,
                timeout_sec=dispatch_task.timeout_sec,
                skip_capacity_wait=True,
                admission_wait_ms_override=0.0,
                admission_snapshot_override=dict(worker_state),
                engine_policy_wait_ms=dispatch_task.engine_policy_wait_ms,
                engine_dispatch_wait_ms=dispatch_task.engine_dispatch_wait_ms,
                enqueue_pending=not self.scheduler_worker.is_engine_decode_control_enabled(),
            )
            dispatch_task.worker_job = worker_job
            self.register_engine_job(worker_job)
            if self.scheduler_worker.is_engine_decode_control_enabled():
                self.decode_runtime_owner.enqueue_pending_job(worker_job)
                self.notify_arbiter()
            self.dispatch_queue_owner.mark_completed(1)
            return True
        except Exception as exc:
            dispatch_task.error = str(exc)
            self.fail_request_state(dispatch_task.engine_request_id or dispatch_task.request_id, str(exc))
            self._notify_dispatch_error(dispatch_task, exc)
            return True
