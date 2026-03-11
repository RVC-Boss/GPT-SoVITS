from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List

import numpy as np
import torch

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SFinishedItem
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import SchedulerFinalizeTask, SchedulerPendingJob


class WorkerFinalizeExecutor:
    def __init__(
        self,
        tts: TTS,
        on_state_change: Callable[[], None] | None = None,
        external_submit: Callable[[List[SchedulerFinalizeTask]], None] | None = None,
    ) -> None:
        self.tts = tts
        self.on_state_change = on_state_change
        self.external_submit = external_submit
        self.condition = threading.Condition()
        self.pending_tasks: Deque[SchedulerFinalizeTask] = deque()
        self.pending_peak = 0
        self.inflight = 0
        self.inflight_peak = 0
        self.worker_count = max(1, int(os.environ.get("GPTSOVITS_FINALIZE_WORKERS", 1)))
        self.finalize_mode = os.environ.get("GPTSOVITS_FINALIZE_MODE", "async").strip().lower()
        self.batch_max_items = max(1, int(os.environ.get("GPTSOVITS_FINALIZE_BATCH_MAX_ITEMS", 16)))
        self.batch_wait_s = max(0.0, float(os.environ.get("GPTSOVITS_FINALIZE_BATCH_WAIT_MS", "2")) / 1000.0)

    def _notify_state_change(self) -> None:
        if self.on_state_change is None:
            return
        try:
            self.on_state_change()
        except Exception:
            pass

    def get_worker_count(self) -> int:
        return int(self.worker_count)

    def get_batch_policy(self) -> Dict[str, Any]:
        return {
            "finalize_mode": str(self.finalize_mode),
            "finalize_batch_max_items": int(self.batch_max_items),
            "finalize_batch_wait_s": float(self.batch_wait_s),
        }

    def get_pending_count(self) -> int:
        with self.condition:
            return int(len(self.pending_tasks))

    def snapshot(self) -> Dict[str, Any]:
        with self.condition:
            return {
                "finalize_pending": int(len(self.pending_tasks)),
                "finalize_pending_peak": int(self.pending_peak),
                "finalize_inflight": int(self.inflight),
                "finalize_inflight_peak": int(self.inflight_peak),
                "finalize_workers": int(self.worker_count),
                "finalize_mode": str(self.finalize_mode),
                "finalize_batch_max_items": int(self.batch_max_items),
                "finalize_batch_wait_ms": float(self.batch_wait_s * 1000.0),
            }

    def is_idle(self) -> bool:
        with self.condition:
            return self.inflight <= 0 and not self.pending_tasks

    def enqueue_tasks(self, tasks: List[SchedulerFinalizeTask]) -> None:
        if not tasks:
            return
        if self.external_submit is not None:
            self.external_submit(tasks)
            self._notify_state_change()
            return
        with self.condition:
            for task in tasks:
                self.pending_tasks.append(task)
            self.pending_peak = max(self.pending_peak, len(self.pending_tasks))
            self.condition.notify_all()
        self._notify_state_change()

    def begin_execution(self, task_count: int) -> None:
        if task_count <= 0:
            return
        with self.condition:
            self.inflight += int(task_count)
            self.inflight_peak = max(self.inflight_peak, self.inflight)
            self.condition.notify_all()
        self._notify_state_change()

    def end_execution(self, task_count: int) -> None:
        with self.condition:
            self.inflight = max(0, self.inflight - int(task_count))
            self.condition.notify_all()
        self._notify_state_change()

    def take_task_batch_blocking(self) -> List[SchedulerFinalizeTask]:
        with self.condition:
            while not self.pending_tasks:
                self.condition.wait()
            selected_tasks = [self.pending_tasks.popleft()]
            if self.finalize_mode == "sync" or self.tts.configs.use_vocoder:
                self.inflight += len(selected_tasks)
                self.inflight_peak = max(self.inflight_peak, self.inflight)
                self._notify_state_change()
                return selected_tasks
            batch_deadline = time.perf_counter() + self.batch_wait_s
            while len(selected_tasks) < self.batch_max_items:
                if not self.pending_tasks:
                    remaining = batch_deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    self.condition.wait(timeout=remaining)
                    continue
                first_task = selected_tasks[0]
                matched_index = None
                for index, task in enumerate(self.pending_tasks):
                    if abs(task.enqueued_time - first_task.enqueued_time) < 1.0:
                        matched_index = index
                        break
                if matched_index is not None:
                    selected_tasks.append(self.pending_tasks[matched_index])
                    del self.pending_tasks[matched_index]
                    continue
                remaining = batch_deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)
            self.inflight += len(selected_tasks)
            self.inflight_peak = max(self.inflight_peak, self.inflight)
        self._notify_state_change()
        return selected_tasks

    def _sync_device(self) -> None:
        try:
            device_str = str(self.tts.configs.device)
            if device_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(self.tts.configs.device)
            elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass

    @staticmethod
    def _collect_job_refer_specs(job: SchedulerPendingJob) -> List[tuple]:
        refer_specs = [job.state.refer_spec]
        refer_specs.extend(list(getattr(job.state, "aux_refer_specs", []) or []))
        return refer_specs

    def _synthesize_finished_audio(self, job: SchedulerPendingJob, item: T2SFinishedItem) -> tuple[int, np.ndarray]:
        audio_fragment = self.tts.synthesize_audio_request_local(
            semantic_tokens=item.semantic_tokens.detach().clone().unsqueeze(0).unsqueeze(0),
            phones=job.state.phones.detach().clone().unsqueeze(0),
            prompt_semantic=job.state.prompt_semantic.detach().clone(),
            prompt_phones=job.state.prompt_phones.detach().clone(),
            refer_spec=[
                (
                    refer_spec_item[0].detach().clone(),
                    None if refer_spec_item[1] is None else refer_spec_item[1].detach().clone(),
                )
                for refer_spec_item in self._collect_job_refer_specs(job)
            ],
            raw_audio=job.state.raw_audio.detach().clone(),
            raw_sr=int(job.state.raw_sr),
            speed=float(job.speed_factor),
            sample_steps=int(job.sample_steps),
        )
        output_sr = self.tts.configs.sampling_rate if not self.tts.configs.use_vocoder else self.tts.vocoder_configs["sr"]
        return self.tts.audio_postprocess(
            audio=[[audio_fragment]],
            sr=int(output_sr),
            batch_index_list=None,
            speed_factor=float(job.speed_factor),
            split_bucket=False,
            fragment_interval=0.0,
            super_sampling=bool(job.super_sampling),
        )

    def _synthesize_finished_audio_batch(
        self,
        jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]],
    ) -> List[tuple[int, np.ndarray]]:
        semantic_tokens_list = [item.semantic_tokens.detach().clone() for _, item in jobs_and_items]
        phones_list = [job.state.phones.detach().clone() for job, _ in jobs_and_items]
        refer_specs = []
        speeds = []
        sample_steps_list = []
        for job, _ in jobs_and_items:
            refer_spec_group = self._collect_job_refer_specs(job)
            if len(refer_spec_group) != 1:
                raise ValueError("batched finalize 暂不支持单请求多参考音频")
            refer_specs.append(
                [(
                    refer_spec_group[0][0].detach().clone(),
                    None if refer_spec_group[0][1] is None else refer_spec_group[0][1].detach().clone(),
                )]
            )
            speeds.append(float(job.speed_factor))
            sample_steps_list.append(int(job.sample_steps))
        audio_fragments = self.tts.synthesize_audio_requests_local_batched(
            semantic_tokens_list=semantic_tokens_list,
            phones_list=phones_list,
            refer_specs=refer_specs,
            speeds=speeds,
            sample_steps_list=sample_steps_list,
        )
        output_sr = self.tts.configs.sampling_rate if not self.tts.configs.use_vocoder else self.tts.vocoder_configs["sr"]
        results: List[tuple[int, np.ndarray]] = []
        for (job, _), audio_fragment in zip(jobs_and_items, audio_fragments):
            results.append(
                self.tts.audio_postprocess(
                    audio=[[audio_fragment]],
                    sr=int(output_sr),
                    batch_index_list=None,
                    speed_factor=float(job.speed_factor),
                    split_bucket=False,
                    fragment_interval=0.0,
                    super_sampling=bool(job.super_sampling),
                )
            )
        return results

    def synthesize_finalize_jobs(
        self,
        jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]],
    ) -> tuple[float, List[tuple[int, np.ndarray]]]:
        if not jobs_and_items:
            return 0.0, []
        self._sync_device()
        synth_start = time.perf_counter()
        if (
            len(jobs_and_items) == 1
            or self.tts.configs.use_vocoder
            or any(len(self._collect_job_refer_specs(job)) != 1 for job, _ in jobs_and_items)
        ):
            batch_results = [self._synthesize_finished_audio(job, item) for job, item in jobs_and_items]
        else:
            batch_results = self._synthesize_finished_audio_batch(jobs_and_items)
        self._sync_device()
        synth_ms = (time.perf_counter() - synth_start) * 1000.0
        return float(synth_ms), batch_results
