from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SActiveBatch, T2SFinishedItem
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineStatus, SchedulerFinalizeTask, SchedulerPendingJob


class WorkerExecutionMixin:
    def execute_prefill_merge(
        self,
        pending_jobs: List[SchedulerPendingJob],
        active_batch: Optional[T2SActiveBatch],
        external_bookkeeping: bool = False,
    ) -> Dict[str, Any]:
        return self.decode_executor.execute_prefill_merge(
            pending_jobs=pending_jobs,
            active_batch=active_batch,
            mark_prefill_started=self._mark_prefill_started,
            add_prefill_time=None if external_bookkeeping else self._add_prefill_time,
            add_merge_time=None if external_bookkeeping else self._add_merge_time,
            enqueue_finished=None if external_bookkeeping else self._enqueue_finalize_finished,
            finalize_error=None if external_bookkeeping else self._finalize_error,
        )

    def execute_decode_step(
        self,
        active_batch: Optional[T2SActiveBatch],
        external_bookkeeping: bool = False,
    ) -> Dict[str, Any]:
        return self.decode_executor.execute_decode_step(
            active_batch=active_batch,
            add_decode_time=None if external_bookkeeping else self._add_decode_time,
            enqueue_finished=None if external_bookkeeping else self._enqueue_finalize_finished,
            finalize_error=None if external_bookkeeping else self._finalize_error,
        )

    def execute_decode_cycle(
        self,
        pending_jobs: List[SchedulerPendingJob],
        active_batch: Optional[T2SActiveBatch],
        external_bookkeeping: bool = False,
    ) -> Dict[str, Any]:
        result = self.decode_executor.execute_decode_cycle(
            pending_jobs=pending_jobs,
            active_batch=active_batch,
            mark_prefill_started=self._mark_prefill_started,
            add_prefill_time=None if external_bookkeeping else self._add_prefill_time,
            add_merge_time=None if external_bookkeeping else self._add_merge_time,
            add_decode_time=None if external_bookkeeping else self._add_decode_time,
            enqueue_finished=None if external_bookkeeping else self._enqueue_finalize_finished,
            finalize_error=None if external_bookkeeping else self._finalize_error,
        )
        self._record_decode_runtime_cycle(result)
        return result

    def run_prefill_merge_once_nonblocking(
        self,
        external_pending_jobs: Optional[List[SchedulerPendingJob]] = None,
        external_active_batch: Optional[T2SActiveBatch] = None,
        emit_runtime_state: bool = True,
        external_bookkeeping: bool = False,
    ) -> Dict[str, Any]:
        result = self.decode_legacy_shell.run_prefill_merge_once_nonblocking(
            external_pending_jobs=external_pending_jobs,
            external_active_batch=external_active_batch,
            execute_prefill_merge=lambda batch_jobs, batch_state: self.execute_prefill_merge(
                pending_jobs=batch_jobs,
                active_batch=batch_state,
                external_bookkeeping=external_bookkeeping,
            ),
        )
        if emit_runtime_state:
            self._notify_decode_runtime_state("prefill_merge")
        return result

    def run_decode_step_once_nonblocking(
        self,
        external_active_batch: Optional[T2SActiveBatch] = None,
        emit_runtime_state: bool = True,
        external_bookkeeping: bool = False,
    ) -> Dict[str, Any]:
        result = self.decode_legacy_shell.run_decode_step_once_nonblocking(
            external_active_batch=external_active_batch,
            execute_decode_step=lambda batch_state: self.execute_decode_step(
                active_batch=batch_state,
                external_bookkeeping=external_bookkeeping,
            ),
        )
        if emit_runtime_state:
            self._notify_decode_runtime_state("decode_step")
        return result

    def run_decode_cycle_nonblocking(
        self,
        external_pending_jobs: Optional[List[SchedulerPendingJob]] = None,
        external_active_batch: Optional[T2SActiveBatch] = None,
        emit_runtime_state: bool = True,
        external_bookkeeping: bool = False,
    ) -> Dict[str, Any]:
        result = self.decode_legacy_shell.run_decode_cycle_nonblocking(
            external_pending_jobs=external_pending_jobs,
            external_active_batch=external_active_batch,
            execute_decode_cycle=lambda batch_jobs, batch_state: self.execute_decode_cycle(
                pending_jobs=batch_jobs,
                active_batch=batch_state,
                external_bookkeeping=external_bookkeeping,
            ),
            on_cycle_executed=None,
        )
        if result.get("executed") and emit_runtime_state:
            self._notify_decode_runtime_state("decode_cycle")
        return result

    def execute_finalize_tasks(self, tasks: List[SchedulerFinalizeTask]) -> None:
        if not tasks:
            return
        try:
            jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]] = []
            with self.condition:
                for task in tasks:
                    job = self.job_registry.get(task.request_id)
                    if job is None:
                        continue
                    jobs_and_items.append((job, task.item))
            if not jobs_and_items:
                return
            now = time.perf_counter()
            for task in tasks:
                self._add_finalize_wait_ms([task.request_id], max(0.0, (now - task.enqueued_time) * 1000.0))
            for job, item in jobs_and_items:
                self._runtime_update(
                    job.engine_request_id,
                    EngineStatus.FINALIZING,
                    {
                        "finish_reason": item.finish_reason,
                        "semantic_len": int(item.semantic_tokens.shape[0]),
                    },
                )
            synth_ms, batch_results = self.synthesize_finalize_jobs(jobs_and_items)
            with self.condition:
                for job, _ in jobs_and_items:
                    tracked_job = self.job_registry.get(job.request_id)
                    if tracked_job is not None:
                        tracked_job.synth_ms += synth_ms
            for (job, item), (sample_rate, audio_data) in zip(jobs_and_items, batch_results):
                self._complete_finalize_task(job, item, sample_rate=sample_rate, audio_data=audio_data)
        except Exception as exc:
            self._finalize_error([task.request_id for task in tasks], str(exc))
        finally:
            self.finalize_executor.end_execution(len(tasks))

    def _run_finalize_loop(self) -> None:
        while True:
            tasks = self.finalize_executor.take_task_batch_blocking()
            self.execute_finalize_tasks(tasks)

    def _run_loop(self) -> None:
        self.decode_legacy_shell.run_loop(
            run_decode_cycle_nonblocking=lambda: self.run_decode_cycle_nonblocking()
        )
