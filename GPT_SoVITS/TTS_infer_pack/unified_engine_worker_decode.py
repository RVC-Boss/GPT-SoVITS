from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import (
    T2SActiveBatch,
    T2SFinishedItem,
    decode_one_step,
    merge_active_batches,
    run_prefill_active_batch,
)
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import RuntimeStateCallbacks, SchedulerPendingJob


class WorkerDecodeExecutor:
    def __init__(self, tts: TTS, max_steps: int) -> None:
        self.tts = tts
        self.max_steps = int(max_steps)

    def _sync_device(self) -> None:
        try:
            device_str = str(self.tts.configs.device)
            if device_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(self.tts.configs.device)
            elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass

    def execute_prefill_merge(
        self,
        *,
        pending_jobs: List[SchedulerPendingJob],
        active_batch: Optional[T2SActiveBatch],
        mark_prefill_started: Callable[[List[SchedulerPendingJob], float], None],
        add_prefill_time: Callable[[List[str], float], None] | None,
        add_merge_time: Callable[[List[str], float], None] | None,
        enqueue_finished: Callable[[List[T2SFinishedItem]], None] | None,
        finalize_error: Callable[[List[str], str], None] | None,
    ) -> Dict[str, Any]:
        if not pending_jobs:
            return {
                "executed": False,
                "active_batch": active_batch,
                "pending_jobs": [],
                "prefill_elapsed_s": 0.0,
                "merge_elapsed_s": 0.0,
                "finished_items": [],
                "error": None,
                "error_request_ids": [],
            }
        admitted_finished: List[T2SFinishedItem] = []
        prefill_elapsed_s = 0.0
        merge_elapsed_s = 0.0
        error: str | None = None
        error_request_ids: List[str] = []
        try:
            self._sync_device()
            prefill_start = time.perf_counter()
            mark_prefill_started(pending_jobs, prefill_start)
            admitted_active_batch, admitted_finished = run_prefill_active_batch(
                self.tts.t2s_model.model,
                [job.state for job in pending_jobs],
                max_steps=self.max_steps,
            )
            self._sync_device()
            prefill_elapsed_s = time.perf_counter() - prefill_start
            if add_prefill_time is not None:
                add_prefill_time([job.request_id for job in pending_jobs], prefill_elapsed_s)
            if enqueue_finished is not None:
                enqueue_finished(admitted_finished)
            merge_start = time.perf_counter()
            active_batch = merge_active_batches(
                self.tts.t2s_model.model,
                active_batch,
                admitted_active_batch,
            )
            merge_elapsed_s = time.perf_counter() - merge_start
            if add_merge_time is not None:
                add_merge_time(
                    [] if active_batch is None else list(active_batch.request_ids),
                    merge_elapsed_s,
                )
        except Exception as exc:
            error = str(exc)
            error_request_ids = [job.request_id for job in pending_jobs]
            if finalize_error is not None:
                finalize_error(error_request_ids, error)
        return {
            "executed": True,
            "active_batch": active_batch,
            "pending_jobs": list(pending_jobs),
            "prefill_elapsed_s": float(prefill_elapsed_s),
            "merge_elapsed_s": float(merge_elapsed_s),
            "finished_items": list(admitted_finished),
            "error": error,
            "error_request_ids": error_request_ids,
        }

    def execute_decode_step(
        self,
        *,
        active_batch: Optional[T2SActiveBatch],
        add_decode_time: Callable[[List[str], float], None] | None,
        enqueue_finished: Callable[[List[T2SFinishedItem]], None] | None,
        finalize_error: Callable[[List[str], str], None] | None,
    ) -> Dict[str, Any]:
        if active_batch is None:
            return {
                "executed": False,
                "active_batch": None,
                "request_ids": [],
                "decode_elapsed_s": 0.0,
                "finished_items": [],
                "error": None,
                "error_request_ids": [],
            }
        active_request_ids: List[str] = []
        step_finished: List[T2SFinishedItem] = []
        decode_elapsed_s = 0.0
        error: str | None = None
        error_request_ids: List[str] = []
        try:
            active_request_ids = [state.request_id for state in active_batch.states]
            self._sync_device()
            decode_start = time.perf_counter()
            active_batch, step_finished = decode_one_step(
                self.tts.t2s_model.model,
                active_batch,
                max_steps=self.max_steps,
            )
            self._sync_device()
            decode_elapsed_s = time.perf_counter() - decode_start
            if add_decode_time is not None:
                add_decode_time(active_request_ids, decode_elapsed_s)
            if enqueue_finished is not None:
                enqueue_finished(step_finished)
        except Exception as exc:
            error = str(exc)
            error_request_ids = list(active_request_ids)
            if finalize_error is not None:
                finalize_error(error_request_ids, error)
            active_batch = None
        return {
            "executed": True,
            "active_batch": active_batch,
            "request_ids": active_request_ids,
            "decode_elapsed_s": float(decode_elapsed_s),
            "finished_items": list(step_finished),
            "error": error,
            "error_request_ids": error_request_ids,
        }

    def execute_decode_cycle(
        self,
        *,
        pending_jobs: List[SchedulerPendingJob],
        active_batch: Optional[T2SActiveBatch],
        mark_prefill_started: Callable[[List[SchedulerPendingJob], float], None],
        add_prefill_time: Callable[[List[str], float], None] | None,
        add_merge_time: Callable[[List[str], float], None] | None,
        add_decode_time: Callable[[List[str], float], None] | None,
        enqueue_finished: Callable[[List[T2SFinishedItem]], None] | None,
        finalize_error: Callable[[List[str], str], None] | None,
    ) -> Dict[str, Any]:
        result = {
            "executed": False,
            "prefill_merge_executed": False,
            "decode_step_executed": False,
            "active_batch": active_batch,
            "prefill_phase": {},
            "decode_phase": {},
        }
        prefill_phase = self.execute_prefill_merge(
            pending_jobs=list(pending_jobs),
            active_batch=result["active_batch"],
            mark_prefill_started=mark_prefill_started,
            add_prefill_time=add_prefill_time,
            add_merge_time=add_merge_time,
            enqueue_finished=enqueue_finished,
            finalize_error=finalize_error,
        )
        prefill_executed = bool(prefill_phase.get("executed", False))
        result["prefill_phase"] = prefill_phase
        result["active_batch"] = prefill_phase.get("active_batch")
        if prefill_executed:
            result["executed"] = True
            result["prefill_merge_executed"] = True
        decode_phase = self.execute_decode_step(
            active_batch=result["active_batch"],
            add_decode_time=add_decode_time,
            enqueue_finished=enqueue_finished,
            finalize_error=finalize_error,
        )
        decode_executed = bool(decode_phase.get("executed", False))
        result["decode_phase"] = decode_phase
        result["active_batch"] = decode_phase.get("active_batch")
        if decode_executed:
            result["executed"] = True
            result["decode_step_executed"] = True
        return result


class WorkerDecodeLegacyShell:
    def __init__(self, condition: threading.Condition, micro_batch_wait_s: float) -> None:
        self.condition = condition
        self.micro_batch_wait_s = float(micro_batch_wait_s)
        self.pending_jobs: List[SchedulerPendingJob] = []
        self.active_batch: T2SActiveBatch | None = None

    @staticmethod
    def _summarize_active_batch(active_batch: T2SActiveBatch | None) -> Dict[str, Any] | None:
        if active_batch is None:
            return None
        return {
            "request_count": int(len(active_batch.request_ids)),
            "request_ids": list(active_batch.request_ids),
            "prefill_done": bool(active_batch.prefill_done),
            "decode_step_index_max": (
                int(active_batch.step_indices.max().item())
                if active_batch.step_indices is not None and active_batch.step_indices.numel() > 0
                else 0
            ),
        }

    def current_backlog_locked(self) -> int:
        running_requests = 0 if self.active_batch is None else len(self.active_batch.request_ids)
        return int(len(self.pending_jobs) + running_requests)

    def enqueue_pending_job_locked(self, job: SchedulerPendingJob) -> None:
        self.pending_jobs.append(job)

    def snapshot_locked(self) -> Dict[str, Any]:
        active_batch_summary = self._summarize_active_batch(self.active_batch)
        executor_local_pending_jobs = int(len(self.pending_jobs))
        executor_local_running_requests = 0 if self.active_batch is None else int(len(self.active_batch.request_ids))
        executor_local_has_work = bool(self.pending_jobs or self.active_batch is not None)
        return {
            "executor_local_pending_jobs": executor_local_pending_jobs,
            "executor_local_running_requests": executor_local_running_requests,
            "executor_local_has_work": executor_local_has_work,
            "executor_local_active_batch": active_batch_summary,
        }

    def is_idle_locked(self) -> bool:
        return self.active_batch is None and not self.pending_jobs

    def take_pending_snapshot(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        with self.condition:
            if not self.pending_jobs and self.active_batch is None:
                self.condition.wait(timeout=self.micro_batch_wait_s)
            elif wait_for_batch and self.pending_jobs:
                self.condition.wait(timeout=self.micro_batch_wait_s)
            if not self.pending_jobs:
                return []
            pending = list(self.pending_jobs)
            self.pending_jobs.clear()
            return pending

    def take_pending_snapshot_nonblocking(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        with self.condition:
            if not self.pending_jobs:
                return []
            if wait_for_batch:
                oldest_enqueue_time = float(self.pending_jobs[0].enqueue_time)
                if (time.perf_counter() - oldest_enqueue_time) < self.micro_batch_wait_s:
                    return []
            pending = list(self.pending_jobs)
            self.pending_jobs.clear()
            return pending

    def has_decode_runtime_work(self) -> bool:
        with self.condition:
            return bool(self.pending_jobs or self.active_batch is not None)

    def build_runtime_summary_locked(self, *, total_cycles: int, prefill_cycles: int, step_cycles: int, last_event: str) -> Dict[str, Any]:
        active_request_ids = [] if self.active_batch is None else list(self.active_batch.request_ids)
        decode_step_index_max = 0
        prefill_done = False
        if self.active_batch is not None:
            prefill_done = bool(self.active_batch.prefill_done)
            if self.active_batch.step_indices is not None and self.active_batch.step_indices.numel() > 0:
                decode_step_index_max = int(self.active_batch.step_indices.max().item())
        return {
            "pending_jobs": int(len(self.pending_jobs)),
            "active_request_count": int(len(active_request_ids)),
            "active_request_ids": active_request_ids[:32],
            "prefill_done": bool(prefill_done),
            "decode_step_index_max": int(decode_step_index_max),
            "total_cycles": int(total_cycles),
            "prefill_cycles": int(prefill_cycles),
            "step_cycles": int(step_cycles),
            "has_work": bool(self.pending_jobs or self.active_batch is not None),
            "last_event": str(last_event),
            "updated_at": float(time.perf_counter()),
        }

    def run_prefill_merge_once_nonblocking(
        self,
        *,
        external_pending_jobs: Optional[List[SchedulerPendingJob]],
        external_active_batch: Optional[T2SActiveBatch],
        execute_prefill_merge: Callable[[List[SchedulerPendingJob], Optional[T2SActiveBatch]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        pending_jobs = (
            list(external_pending_jobs)
            if external_pending_jobs is not None
            else self.take_pending_snapshot_nonblocking(wait_for_batch=self.active_batch is None)
        )
        active_batch = external_active_batch if external_pending_jobs is not None else self.active_batch
        result = execute_prefill_merge(pending_jobs, active_batch)
        if external_pending_jobs is None:
            with self.condition:
                self.active_batch = result.get("active_batch")
                self.condition.notify_all()
        return result

    def run_decode_step_once_nonblocking(
        self,
        *,
        external_active_batch: Optional[T2SActiveBatch],
        execute_decode_step: Callable[[Optional[T2SActiveBatch]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        active_batch = self.active_batch if external_active_batch is None else external_active_batch
        result = execute_decode_step(active_batch)
        if external_active_batch is None:
            with self.condition:
                self.active_batch = result.get("active_batch")
                self.condition.notify_all()
        return result

    def run_decode_cycle_nonblocking(
        self,
        *,
        external_pending_jobs: Optional[List[SchedulerPendingJob]],
        external_active_batch: Optional[T2SActiveBatch],
        execute_decode_cycle: Callable[[List[SchedulerPendingJob], Optional[T2SActiveBatch]], Dict[str, Any]],
        on_cycle_executed: Callable[[Dict[str, Any]], None] | None,
    ) -> Dict[str, Any]:
        pending_jobs = (
            list(external_pending_jobs)
            if external_pending_jobs is not None
            else self.take_pending_snapshot_nonblocking(wait_for_batch=self.active_batch is None)
        )
        active_batch = external_active_batch if external_pending_jobs is not None else self.active_batch
        result = execute_decode_cycle(pending_jobs, active_batch)
        if external_pending_jobs is None:
            with self.condition:
                self.active_batch = result.get("active_batch")
                self.condition.notify_all()
        if result.get("executed") and on_cycle_executed is not None:
            on_cycle_executed(result)
        return result

    def run_loop(
        self,
        *,
        run_decode_cycle_nonblocking: Callable[[], Dict[str, Any]],
    ) -> None:
        while True:
            executed = run_decode_cycle_nonblocking()
            if executed.get("executed"):
                continue
            wait_for_batch = self.active_batch is None
            pending_jobs = self.take_pending_snapshot(wait_for_batch=wait_for_batch)
            if pending_jobs:
                with self.condition:
                    self.pending_jobs = pending_jobs + self.pending_jobs
                    self.condition.notify_all()
                continue
            time.sleep(self.micro_batch_wait_s)


class WorkerDecodeRuntimeTracker:
    def __init__(
        self,
        runtime_callbacks: RuntimeStateCallbacks | None = None,
    ) -> None:
        self.runtime_callbacks = runtime_callbacks or RuntimeStateCallbacks()
        self.total_cycles = 0
        self.prefill_cycles = 0
        self.step_cycles = 0

    def get_counters(self) -> Dict[str, int]:
        return {
            "total_cycles": int(self.total_cycles),
            "prefill_cycles": int(self.prefill_cycles),
            "step_cycles": int(self.step_cycles),
        }

    def record_cycle(self, result: Dict[str, Any]) -> None:
        if not bool(result.get("executed")):
            return
        self.total_cycles += 1
        if bool(result.get("prefill_merge_executed")):
            self.prefill_cycles += 1
        if bool(result.get("decode_step_executed")):
            self.step_cycles += 1

    def build_runtime_summary_locked(
        self,
        *,
        legacy_shell: WorkerDecodeLegacyShell,
        last_event: str,
    ) -> Dict[str, Any]:
        return legacy_shell.build_runtime_summary_locked(
            total_cycles=int(self.total_cycles),
            prefill_cycles=int(self.prefill_cycles),
            step_cycles=int(self.step_cycles),
            last_event=str(last_event),
        )

    def notify_runtime_update_locked(
        self,
        *,
        legacy_shell: WorkerDecodeLegacyShell,
        last_event: str,
    ) -> None:
        if self.runtime_callbacks.decode_runtime_update is None:
            return
        snapshot = self.build_runtime_summary_locked(
            legacy_shell=legacy_shell,
            last_event=last_event,
        )
        self.runtime_callbacks.decode_runtime_update(snapshot)
