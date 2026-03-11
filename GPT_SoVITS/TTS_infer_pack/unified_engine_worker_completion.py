from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SFinishedItem
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import RuntimeStateCallbacks, SchedulerJobRegistry, SchedulerPendingJob


class WorkerCompletionBridge:
    def __init__(self, runtime_callbacks: RuntimeStateCallbacks | None = None) -> None:
        self.runtime_callbacks = runtime_callbacks or RuntimeStateCallbacks()

    @staticmethod
    def _resolve_done_future(job: SchedulerPendingJob) -> None:
        future = job.done_future
        if future is None or future.done():
            return
        future.set_result(job)

    def notify_done_future(self, job: SchedulerPendingJob) -> None:
        if job.done_loop is None or job.done_future is None:
            return
        try:
            job.done_loop.call_soon_threadsafe(self._resolve_done_future, job)
        except RuntimeError:
            pass

    def runtime_complete(self, request_id: str | None, extra: Optional[Dict[str, Any]] = None) -> None:
        if request_id is None or self.runtime_callbacks.complete is None:
            return
        self.runtime_callbacks.complete(request_id, extra)

    def runtime_fail(self, request_id: str | None, error: str) -> None:
        if request_id is None or self.runtime_callbacks.fail is None:
            return
        self.runtime_callbacks.fail(request_id, error)

    @staticmethod
    def build_completed_job_result(
        job: SchedulerPendingJob,
        item: T2SFinishedItem,
        *,
        sample_rate: int,
        audio_data: np.ndarray,
        finished_at: float | None = None,
    ) -> Dict[str, Any]:
        finished_at = float(time.perf_counter() if finished_at is None else finished_at)
        queue_wait_ms = 0.0
        if job.first_schedule_time is not None:
            queue_wait_ms = max(0.0, (job.first_schedule_time - job.enqueue_time) * 1000.0)
        worker_total_ms = max(0.0, (finished_at - job.enqueue_time) * 1000.0)
        worker_residual_ms = max(
            0.0,
            worker_total_ms
            - queue_wait_ms
            - job.prefill_ms
            - job.merge_ms
            - job.decode_ms
            - job.finalize_wait_ms
            - job.synth_ms,
        )
        worker_other_ms = max(0.0, job.merge_ms + job.finalize_wait_ms + worker_residual_ms)
        job.sample_rate = int(sample_rate)
        job.audio_data = audio_data
        job.result_ready_time = finished_at
        prepare_profile = dict(job.state.prepare_profile)
        result = {
            "request_id": item.request_id,
            "semantic_len": int(item.semantic_tokens.shape[0]),
            "finish_idx": int(item.finish_idx),
            "finish_reason": item.finish_reason,
            "decode_admission_wait_ms": float(job.admission_wait_ms),
            "engine_policy_wait_ms": float(job.engine_policy_wait_ms),
            "engine_dispatch_wait_ms": float(job.engine_dispatch_wait_ms),
            "prepare_ms": job.prepare_wall_ms,
            "prepare_wall_ms": job.prepare_wall_ms,
            "prepare_profile_total_ms": job.prepare_profile_total_ms,
            "prepare_profile": prepare_profile,
            "queue_wait_ms": queue_wait_ms,
            "prefill_ms": job.prefill_ms,
            "merge_ms": job.merge_ms,
            "decode_ms": job.decode_ms,
            "finalize_wait_ms": job.finalize_wait_ms,
            "synth_ms": job.synth_ms,
            "worker_residual_ms": worker_residual_ms,
            "worker_other_ms": worker_other_ms,
            "worker_total_ms": worker_total_ms,
            "decode_steps": int(job.decode_steps),
            "sample_rate": int(sample_rate),
            "media_type": job.media_type,
        }
        job.result = result
        return result

    @staticmethod
    def build_runtime_complete_payload(
        job: SchedulerPendingJob,
        item: T2SFinishedItem,
        *,
        sample_rate: int,
    ) -> Dict[str, Any]:
        return {
            "finish_reason": item.finish_reason,
            "semantic_len": int(item.semantic_tokens.shape[0]),
            "finish_idx": int(item.finish_idx),
            "sample_rate": int(sample_rate),
            "worker_profile": dict(job.result or {}),
        }

    def complete_job(
        self,
        job: SchedulerPendingJob,
        *,
        runtime_request_id: str | None,
        runtime_extra: Optional[Dict[str, Any]] = None,
        remove_job: Callable[[], None] | None = None,
        on_job_finished: Callable[[], None] | None = None,
        notify_waiters: Callable[[], None] | None = None,
    ) -> None:
        job.done_event.set()
        self.notify_done_future(job)
        if remove_job is not None:
            remove_job()
        if on_job_finished is not None:
            on_job_finished()
        if notify_waiters is not None:
            notify_waiters()
        self.runtime_complete(runtime_request_id, runtime_extra)

    def fail_job(
        self,
        job: SchedulerPendingJob,
        *,
        error: str,
        remove_job: Callable[[], None] | None = None,
        on_job_finished: Callable[[], None] | None = None,
        notify_waiters: Callable[[], None] | None = None,
    ) -> None:
        job.error = str(error)
        job.done_event.set()
        self.notify_done_future(job)
        if remove_job is not None:
            remove_job()
        if on_job_finished is not None:
            on_job_finished()
        if notify_waiters is not None:
            notify_waiters()
        self.runtime_fail(job.engine_request_id, str(error))

    def complete_finalize_task(
        self,
        *,
        condition: threading.Condition,
        job_registry: SchedulerJobRegistry,
        job: SchedulerPendingJob,
        item: T2SFinishedItem,
        sample_rate: int,
        audio_data: np.ndarray,
    ) -> None:
        runtime_extra: Optional[Dict[str, Any]] = None
        with condition:
            if job_registry.get(item.request_id) is not job:
                return
            self.build_completed_job_result(job, item, sample_rate=sample_rate, audio_data=audio_data)
            runtime_extra = self.build_runtime_complete_payload(job, item, sample_rate=sample_rate)
            self.complete_job(
                job,
                runtime_request_id=job.engine_request_id,
                runtime_extra=runtime_extra,
                on_job_finished=lambda: job_registry.mark_finished_and_remove(item.request_id),
                notify_waiters=condition.notify_all,
            )

    def fail_jobs(
        self,
        *,
        condition: threading.Condition,
        job_registry: SchedulerJobRegistry,
        request_ids: List[str],
        error: str,
    ) -> None:
        if not request_ids:
            return
        with condition:
            for request_id in request_ids:
                job = job_registry.get(request_id)
                if job is None:
                    continue
                self.fail_job(
                    job,
                    error=error,
                    on_job_finished=lambda rid=request_id: job_registry.mark_finished_and_remove(rid),
                )
            condition.notify_all()
