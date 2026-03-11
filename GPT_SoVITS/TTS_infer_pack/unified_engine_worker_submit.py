from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List

from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import PreparedCpuStage
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineStatus, SchedulerPendingJob


class WorkerSubmitLifecycleMixin:
    def _current_decode_backlog_locked(self) -> int:
        return self.decode_legacy_shell.current_backlog_locked()

    def get_micro_batch_wait_s(self) -> float:
        return float(self.micro_batch_wait_s)

    def is_engine_decode_control_enabled(self) -> bool:
        return bool(self.engine_decode_control_enabled)

    def get_prepare_max_inflight(self) -> int:
        return int(self.prepare_executor.get_max_inflight())

    def get_capacity_limits(self) -> Dict[str, int]:
        return {
            "decode_backlog_max": int(self.decode_backlog_max),
            "finalize_pending_max": int(self.finalize_pending_max),
        }

    def get_finalize_batch_policy(self) -> Dict[str, Any]:
        return dict(self.finalize_executor.get_batch_policy())

    def get_decode_runtime_counters(self) -> Dict[str, int]:
        with self.condition:
            return self.decode_runtime_tracker.get_counters()

    def _can_accept_submit_locked(self) -> tuple[bool, Dict[str, int]]:
        decode_backlog = self._current_decode_backlog_locked()
        finalize_pending = int(self.finalize_executor.get_pending_count())
        prepare_inflight = int(self.prepare_executor.snapshot()["inflight"])
        blocked_decode = self.decode_backlog_max > 0 and decode_backlog >= self.decode_backlog_max
        blocked_finalize = self.finalize_pending_max > 0 and finalize_pending >= self.finalize_pending_max
        return (
            not blocked_decode and not blocked_finalize,
            {
                "decode_backlog": decode_backlog,
                "finalize_pending": finalize_pending,
                "prepare_inflight": prepare_inflight,
                "decode_backlog_max": int(self.decode_backlog_max),
                "finalize_pending_max": int(self.finalize_pending_max),
            },
        )

    def wait_for_submit_capacity_blocking(self, timeout_sec: float | None = None) -> tuple[float, Dict[str, int]]:
        start = time.perf_counter()
        deadline = None if timeout_sec in [None, ""] else (start + max(0.0, float(timeout_sec)))
        while True:
            with self.condition:
                allowed, snapshot = self._can_accept_submit_locked()
                if allowed:
                    return max(0.0, (time.perf_counter() - start) * 1000.0), snapshot
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError(
                        "scheduler submit admission timeout "
                        f"(decode_backlog={snapshot['decode_backlog']}, finalize_pending={snapshot['finalize_pending']})"
                    )
                self.condition.wait(timeout=self.micro_batch_wait_s)

    def _admission_snapshot_locked(self) -> Dict[str, int]:
        _, snapshot = self._can_accept_submit_locked()
        return snapshot

    async def submit_async(
        self,
        state: T2SRequestState,
        speed_factor: float,
        sample_steps: int,
        media_type: str,
        super_sampling: bool,
        prepare_wall_ms: float,
        prepare_profile_total_ms: float,
        done_loop: asyncio.AbstractEventLoop | None = None,
        done_future: asyncio.Future | None = None,
        engine_request_id: str | None = None,
        timeout_sec: float | None = None,
        skip_capacity_wait: bool = False,
        admission_wait_ms_override: float | None = None,
        admission_snapshot_override: Dict[str, Any] | None = None,
        engine_policy_wait_ms: float = 0.0,
        engine_dispatch_wait_ms: float = 0.0,
        enqueue_pending: bool = True,
    ) -> SchedulerPendingJob:
        return await asyncio.to_thread(
            self.submit,
            state,
            speed_factor,
            sample_steps,
            media_type,
            super_sampling,
            prepare_wall_ms,
            prepare_profile_total_ms,
            done_loop,
            done_future,
            engine_request_id,
            timeout_sec,
            skip_capacity_wait,
            admission_wait_ms_override,
            admission_snapshot_override,
            engine_policy_wait_ms,
            engine_dispatch_wait_ms,
            enqueue_pending,
        )

    def snapshot(self) -> dict:
        with self.condition:
            prepare_state = self.prepare_executor.snapshot()
            finalize_state = self.finalize_executor.snapshot()
            shell_state = self.decode_legacy_shell.snapshot_locked()
            decode_runtime_counters = self.decode_runtime_tracker.get_counters()
            engine_owned_decode_state = bool(self.engine_decode_control_enabled)
            active_batch_summary = shell_state.get("executor_local_active_batch")
            executor_local_pending_jobs = int(shell_state.get("executor_local_pending_jobs", 0))
            executor_local_running_requests = int(shell_state.get("executor_local_running_requests", 0))
            executor_local_has_work = bool(shell_state.get("executor_local_has_work", False))
            return {
                "pending_jobs": 0 if engine_owned_decode_state else executor_local_pending_jobs,
                "running_requests": 0 if engine_owned_decode_state else executor_local_running_requests,
                "engine_decode_control_enabled": bool(self.engine_decode_control_enabled),
                "legacy_state_owner_mode": not engine_owned_decode_state,
                "decode_state_owner": "engine" if engine_owned_decode_state else "worker",
                "decode_runtime_has_work": False if engine_owned_decode_state else executor_local_has_work,
                "executor_local_pending_jobs": executor_local_pending_jobs,
                "executor_local_running_requests": executor_local_running_requests,
                "executor_local_has_work": executor_local_has_work,
                "decode_runtime_total_cycles": int(decode_runtime_counters.get("total_cycles", 0)),
                "decode_runtime_prefill_cycles": int(decode_runtime_counters.get("prefill_cycles", 0)),
                "decode_runtime_step_cycles": int(decode_runtime_counters.get("step_cycles", 0)),
                "prepare_inflight": prepare_state["inflight"],
                "prepare_peak_inflight": prepare_state["peak_inflight"],
                "prepare_max_inflight": prepare_state.get("max_inflight", 0),
                "prepare_state": dict(prepare_state),
                **finalize_state,
                "decode_backlog_max": self.decode_backlog_max,
                "finalize_pending_max": self.finalize_pending_max,
                "active_batch": {} if engine_owned_decode_state else active_batch_summary,
                "executor_local_active_batch": active_batch_summary if engine_owned_decode_state else None,
                "total_submitted": self.job_registry.submitted_count(),
                "total_finished": self.job_registry.finished_count(),
                "drained": self.is_drained(),
            }

    def is_drained(self) -> bool:
        with self.condition:
            return (
                self.decode_legacy_shell.is_idle_locked()
                and self.job_registry.is_empty()
                and self.prepare_executor.is_idle()
                and self.finalize_executor.is_idle()
            )

    def wait_until_idle(self, timeout_sec: float = 60.0, poll_interval_sec: float = 0.01) -> bool:
        deadline = time.perf_counter() + max(0.0, timeout_sec)
        while time.perf_counter() < deadline:
            if self.is_drained():
                return True
            time.sleep(poll_interval_sec)
        return self.is_drained()

    def submit(
        self,
        state: T2SRequestState,
        speed_factor: float,
        sample_steps: int,
        media_type: str,
        super_sampling: bool,
        prepare_wall_ms: float,
        prepare_profile_total_ms: float,
        done_loop: asyncio.AbstractEventLoop | None = None,
        done_future: asyncio.Future | None = None,
        engine_request_id: str | None = None,
        timeout_sec: float | None = None,
        skip_capacity_wait: bool = False,
        admission_wait_ms_override: float | None = None,
        admission_snapshot_override: Dict[str, Any] | None = None,
        engine_policy_wait_ms: float = 0.0,
        engine_dispatch_wait_ms: float = 0.0,
        enqueue_pending: bool = True,
    ) -> SchedulerPendingJob:
        if skip_capacity_wait:
            with self.condition:
                admission_snapshot = (
                    dict(admission_snapshot_override)
                    if admission_snapshot_override is not None
                    else dict(self._admission_snapshot_locked())
                )
            admission_wait_ms = 0.0 if admission_wait_ms_override is None else float(admission_wait_ms_override)
        else:
            admission_wait_ms, admission_snapshot = self.wait_for_submit_capacity_blocking(timeout_sec=timeout_sec)
        job = SchedulerPendingJob(
            request_id=state.request_id,
            state=state,
            done_event=threading.Event(),
            done_loop=done_loop,
            done_future=done_future,
            enqueue_time=time.perf_counter(),
            speed_factor=float(speed_factor),
            sample_steps=int(sample_steps),
            media_type=media_type,
            super_sampling=bool(super_sampling),
            admission_wait_ms=float(admission_wait_ms),
            engine_policy_wait_ms=float(engine_policy_wait_ms),
            engine_dispatch_wait_ms=float(engine_dispatch_wait_ms),
            prepare_wall_ms=float(prepare_wall_ms),
            prepare_profile_total_ms=float(prepare_profile_total_ms),
            engine_request_id=engine_request_id or state.request_id,
        )
        with self.condition:
            self.job_registry.register(job, keep_job=not self.engine_decode_control_enabled)
            if enqueue_pending:
                self.decode_legacy_shell.enqueue_pending_job_locked(job)
            self.condition.notify_all()
        if enqueue_pending:
            self._notify_decode_runtime_state("submit")
        self._runtime_update(
            job.engine_request_id,
            EngineStatus.QUEUED,
            {
                "scheduler_request_id": job.request_id,
                "decode_admission_wait_ms": float(admission_wait_ms),
                "engine_policy_wait_ms": float(engine_policy_wait_ms),
                "engine_dispatch_wait_ms": float(engine_dispatch_wait_ms),
                "admission_snapshot": dict(admission_snapshot),
            },
        )
        return job

    async def prepare_state_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> tuple[T2SRequestState, float, float]:
        return await self.prepare_executor.prepare_state_profiled_async(spec, prepare_submit_at)

    async def prepare_states_batch_async(self, specs: List[SchedulerRequestSpec]) -> List[T2SRequestState]:
        return await self.prepare_executor.prepare_states_batch_async(specs)

    async def prepare_cpu_stage_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> PreparedCpuStage:
        return await self.prepare_executor.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)

    async def prepare_gpu_stage_profiled_async(
        self,
        cpu_stage: PreparedCpuStage,
    ) -> tuple[T2SRequestState, float, float]:
        return await self.prepare_executor.prepare_gpu_stage_profiled_async(cpu_stage)
