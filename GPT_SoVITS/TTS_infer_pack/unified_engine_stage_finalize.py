from __future__ import annotations

import time
from typing import List

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SFinishedItem
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineStatus, SchedulerFinalizeTask, SchedulerPendingJob


class EngineFinalizeStageMixin:
    def enqueue_worker_finished_for_finalize(self, tasks: List[SchedulerFinalizeTask]) -> None:
        if not tasks:
            return
        for task in tasks:
            job = self.get_engine_job(task.request_id)
            if job is not None:
                self.update_request_state(
                    job.engine_request_id,
                    EngineStatus.READY_FOR_FINALIZE,
                    {
                        "finish_reason": task.item.finish_reason,
                        "semantic_len": int(task.item.semantic_tokens.shape[0]),
                        "finish_idx": int(task.item.finish_idx),
                    },
                )
        self.finalize_queue_owner.enqueue_many(tasks)
        self.notify_arbiter()

    def take_engine_finalize_batch_nonblocking(self) -> List[SchedulerFinalizeTask]:
        finalize_policy = self.scheduler_worker.get_finalize_batch_policy()
        return self.finalize_queue_owner.take_finalize_batch(
            finalize_mode=str(finalize_policy.get("finalize_mode", "async")),
            batch_max_items=int(finalize_policy.get("finalize_batch_max_items", 1)),
            batch_wait_s=float(finalize_policy.get("finalize_batch_wait_s", 0.0)),
            use_vocoder=bool(self.tts.configs.use_vocoder),
        )

    def run_engine_finalize_once(self) -> bool:
        tasks = self.take_engine_finalize_batch_nonblocking()
        if not tasks:
            return False
        ready_tasks: List[SchedulerFinalizeTask] = []
        failed_tasks: List[SchedulerFinalizeTask] = []
        deferred_tasks: List[SchedulerFinalizeTask] = []
        for task in tasks:
            job = self.get_engine_job(task.request_id)
            if job is None:
                continue
            if float(job.state.prepare_profile.get("ref_spec_async_failed", 0.0) or 0.0) > 0.0:
                failed_tasks.append(task)
                continue
            if job.state.refer_spec is None:
                deferred_tasks.append(task)
                self.merge_request_state_profile(
                    job.engine_request_id or job.request_id,
                    {
                        "engine_finalize_ref_spec_blocked": 1.0,
                    },
                )
                continue
            ready_tasks.append(task)
        if deferred_tasks:
            self.finalize_queue_owner.enqueue_many(deferred_tasks)
        if failed_tasks:
            self.fail_engine_jobs([task.request_id for task in failed_tasks], "ref_spec async stage failed")
        if not ready_tasks:
            self.finalize_queue_owner.mark_completed(len(failed_tasks), notify=True)
            return False
        self.scheduler_worker.begin_finalize_execution(len(ready_tasks))
        try:
            jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]] = []
            for task in ready_tasks:
                job = self.get_engine_job(task.request_id)
                if job is None:
                    continue
                jobs_and_items.append((job, task.item))
            if not jobs_and_items:
                return False
            now = time.perf_counter()
            for task in ready_tasks:
                job = self.get_engine_job(task.request_id)
                if job is not None:
                    job.finalize_wait_ms += max(0.0, (now - task.enqueued_time) * 1000.0)
            for job, item in jobs_and_items:
                self.update_request_state(
                    job.engine_request_id,
                    EngineStatus.FINALIZING,
                    {
                        "finish_reason": item.finish_reason,
                        "semantic_len": int(item.semantic_tokens.shape[0]),
                    },
                )
            synth_ms, batch_results = self.scheduler_worker.synthesize_finalize_jobs(jobs_and_items)
            for job, _ in jobs_and_items:
                job.synth_ms += float(synth_ms)
            for (job, item), (sample_rate, audio_data) in zip(jobs_and_items, batch_results):
                self.complete_engine_job(job, item, sample_rate=sample_rate, audio_data=audio_data)
        except Exception as exc:
            self.fail_engine_jobs([task.request_id for task in ready_tasks], str(exc))
        finally:
            self.scheduler_worker.end_finalize_execution(len(ready_tasks))
        self.finalize_queue_owner.mark_completed(len(ready_tasks) + len(failed_tasks), notify=True)
        return True
