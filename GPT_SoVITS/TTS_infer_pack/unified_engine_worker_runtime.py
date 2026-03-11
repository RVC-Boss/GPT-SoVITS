from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SFinishedItem
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineStatus, SchedulerFinalizeTask, SchedulerPendingJob


class WorkerRuntimeBookkeepingMixin:
    def _mark_prefill_started(self, pending_jobs: List[SchedulerPendingJob], started_at: float) -> None:
        with self.condition:
            for job in pending_jobs:
                job.first_schedule_time = float(started_at)
                self._runtime_update(
                    job.engine_request_id,
                    EngineStatus.GPU_PREPARING,
                    {"scheduler_request_id": job.request_id, "prefill_started_at": float(started_at)},
                )

    def _add_prefill_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        if not request_ids:
            return
        with self.condition:
            for request_id in request_ids:
                job = self.job_registry.get(request_id)
                if job is not None:
                    job.prefill_ms += delta_ms

    def _add_merge_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        if not request_ids:
            return
        with self.condition:
            for request_id in request_ids:
                job = self.job_registry.get(request_id)
                if job is not None:
                    job.merge_ms += delta_ms

    def _add_decode_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        if not request_ids:
            return
        activate_request_ids: List[str] = []
        with self.condition:
            for request_id in request_ids:
                job = self.job_registry.get(request_id)
                if job is not None:
                    if job.decode_steps == 0:
                        activate_request_ids.append(job.engine_request_id)
                    job.decode_ms += delta_ms
                    job.decode_steps += 1
        for engine_request_id in activate_request_ids:
            self._runtime_update(engine_request_id, EngineStatus.ACTIVE_DECODE, None)

    def _add_finalize_wait_ms(self, request_ids: List[str], delta_ms: float) -> None:
        if not request_ids:
            return
        with self.condition:
            for request_id in request_ids:
                job = self.job_registry.get(request_id)
                if job is not None:
                    job.finalize_wait_ms += float(delta_ms)

    def _enqueue_finalize_finished(self, items: List[T2SFinishedItem]) -> None:
        if not items:
            return
        enqueued_at = time.perf_counter()
        tasks: List[SchedulerFinalizeTask] = []
        with self.condition:
            for item in items:
                job = self.job_registry.get(item.request_id)
                if job is not None:
                    self._runtime_update(
                        job.engine_request_id,
                        EngineStatus.READY_FOR_FINALIZE,
                        {
                            "finish_reason": item.finish_reason,
                            "semantic_len": int(item.semantic_tokens.shape[0]),
                            "finish_idx": int(item.finish_idx),
                        },
                    )
                tasks.append(SchedulerFinalizeTask(request_id=item.request_id, item=item, enqueued_time=enqueued_at))
        self.finalize_executor.enqueue_tasks(tasks)

    def begin_finalize_execution(self, task_count: int) -> None:
        self.finalize_executor.begin_execution(task_count)

    def end_finalize_execution(self, task_count: int) -> None:
        self.finalize_executor.end_execution(task_count)

    def record_external_job_done(self, request_id: str) -> None:
        with self.condition:
            self.job_registry.mark_finished_and_remove(request_id)
            self.condition.notify_all()

    def synthesize_finalize_jobs(
        self,
        jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]],
    ) -> tuple[float, List[tuple[int, np.ndarray]]]:
        return self.finalize_executor.synthesize_finalize_jobs(jobs_and_items)

    def _complete_finalize_task(self, job: SchedulerPendingJob, item: T2SFinishedItem, sample_rate: int, audio_data: np.ndarray) -> None:
        self.completion_bridge.complete_finalize_task(
            condition=self.condition,
            job_registry=self.job_registry,
            job=job,
            item=item,
            sample_rate=sample_rate,
            audio_data=audio_data,
        )

    def _finalize_error(self, request_ids: List[str], error: str) -> None:
        self.completion_bridge.fail_jobs(
            condition=self.condition,
            job_registry=self.job_registry,
            request_ids=request_ids,
            error=error,
        )

    @staticmethod
    def _resolve_done_future(job: SchedulerPendingJob) -> None:
        future = job.done_future
        if future is None or future.done():
            return
        future.set_result(job)

    def _notify_done_future(self, job: SchedulerPendingJob) -> None:
        self.completion_bridge.notify_done_future(job)

    def _runtime_update(self, request_id: str | None, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if request_id is None or self.runtime_callbacks.update is None:
            return
        self.runtime_callbacks.update(request_id, status, extra)

    def _runtime_complete(self, request_id: str | None, extra: Optional[Dict[str, Any]] = None) -> None:
        self.completion_bridge.runtime_complete(request_id, extra)

    def _runtime_fail(self, request_id: str | None, error: str) -> None:
        self.completion_bridge.runtime_fail(request_id, error)

    def _build_decode_runtime_summary_locked(self, last_event: str) -> Dict[str, Any]:
        return self.decode_runtime_tracker.build_runtime_summary_locked(
            legacy_shell=self.decode_legacy_shell,
            last_event=str(last_event),
        )

    def _notify_decode_runtime_state(self, last_event: str) -> None:
        with self.condition:
            self.decode_runtime_tracker.notify_runtime_update_locked(
                legacy_shell=self.decode_legacy_shell,
                last_event=str(last_event),
            )

    def _record_decode_runtime_cycle(self, result: Dict[str, Any]) -> None:
        with self.condition:
            self.decode_runtime_tracker.record_cycle(result)

    def _take_pending_snapshot(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        return self.decode_legacy_shell.take_pending_snapshot(wait_for_batch)

    def _take_pending_snapshot_nonblocking(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        return self.decode_legacy_shell.take_pending_snapshot_nonblocking(wait_for_batch)

    def has_decode_runtime_work(self) -> bool:
        return self.decode_legacy_shell.has_decode_runtime_work()
