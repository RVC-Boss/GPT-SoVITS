from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import PrepareCoordinator
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import (
    SchedulerRequestSpec,
    T2SActiveBatch,
    T2SFinishedItem,
    T2SRequestState,
    decode_one_step,
    merge_active_batches,
    run_prefill_active_batch,
    run_scheduler_continuous,
)


@dataclass
class RuntimeControlCallbacks:
    restart: Callable[[], None] | None = None
    exit: Callable[[], None] | None = None


@dataclass
class DefaultReferenceState:
    ref_audio_path: str | None = None
    updated_at: float = 0.0


class ReferenceRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = DefaultReferenceState()

    def set_default(self, ref_audio_path: str) -> DefaultReferenceState:
        with self._lock:
            self._state = DefaultReferenceState(ref_audio_path=str(ref_audio_path), updated_at=time.time())
            return self._state

    def clear(self) -> DefaultReferenceState:
        with self._lock:
            self._state = DefaultReferenceState()
            return self._state

    def get_default(self) -> DefaultReferenceState:
        with self._lock:
            return DefaultReferenceState(
                ref_audio_path=self._state.ref_audio_path,
                updated_at=self._state.updated_at,
            )


@dataclass
class ModelRegistryState:
    t2s_weights_path: str
    vits_weights_path: str
    generation: int = 0
    t2s_generation: int = 0
    vits_generation: int = 0
    updated_at: float = field(default_factory=time.time)


class ModelRegistry:
    def __init__(self, t2s_weights_path: str, vits_weights_path: str) -> None:
        self._lock = threading.Lock()
        self._state = ModelRegistryState(
            t2s_weights_path=str(t2s_weights_path),
            vits_weights_path=str(vits_weights_path),
        )

    def snapshot(self) -> ModelRegistryState:
        with self._lock:
            return ModelRegistryState(
                t2s_weights_path=self._state.t2s_weights_path,
                vits_weights_path=self._state.vits_weights_path,
                generation=self._state.generation,
                t2s_generation=self._state.t2s_generation,
                vits_generation=self._state.vits_generation,
                updated_at=self._state.updated_at,
            )

    def mark_t2s_reload(self, weights_path: str) -> ModelRegistryState:
        with self._lock:
            self._state.t2s_weights_path = str(weights_path)
            self._state.generation += 1
            self._state.t2s_generation += 1
            self._state.updated_at = time.time()
            return ModelRegistryState(
                t2s_weights_path=self._state.t2s_weights_path,
                vits_weights_path=self._state.vits_weights_path,
                generation=self._state.generation,
                t2s_generation=self._state.t2s_generation,
                vits_generation=self._state.vits_generation,
                updated_at=self._state.updated_at,
            )

    def mark_vits_reload(self, weights_path: str) -> ModelRegistryState:
        with self._lock:
            self._state.vits_weights_path = str(weights_path)
            self._state.generation += 1
            self._state.vits_generation += 1
            self._state.updated_at = time.time()
            return ModelRegistryState(
                t2s_weights_path=self._state.t2s_weights_path,
                vits_weights_path=self._state.vits_weights_path,
                generation=self._state.generation,
                t2s_generation=self._state.t2s_generation,
                vits_generation=self._state.vits_generation,
                updated_at=self._state.updated_at,
            )


@dataclass
class DirectTTSExecution:
    media_type: str
    streaming: bool
    audio_generator: Optional[Generator[bytes, None, None]] = None
    audio_bytes: Optional[bytes] = None


@dataclass
class SchedulerDebugExecution:
    payload: Dict[str, Any]


@dataclass
class SchedulerSubmitExecution:
    audio_bytes: bytes
    media_type: str
    headers: Dict[str, str]


@dataclass
class SchedulerPendingJob:
    request_id: str
    state: T2SRequestState
    done_event: threading.Event
    done_loop: asyncio.AbstractEventLoop | None
    done_future: asyncio.Future | None
    enqueue_time: float
    speed_factor: float
    sample_steps: int
    media_type: str
    prepare_wall_ms: float = 0.0
    prepare_profile_total_ms: float = 0.0
    first_schedule_time: float | None = None
    prefill_ms: float = 0.0
    merge_ms: float = 0.0
    decode_ms: float = 0.0
    finalize_wait_ms: float = 0.0
    synth_ms: float = 0.0
    pack_ms: float = 0.0
    decode_steps: int = 0
    result_ready_time: float | None = None
    result: dict | None = None
    sample_rate: int | None = None
    audio_data: np.ndarray | None = None
    error: str | None = None


@dataclass
class SchedulerFinalizeTask:
    request_id: str
    item: T2SFinishedItem
    enqueued_time: float


class UnifiedSchedulerWorker:
    def __init__(self, tts: TTS, max_steps: int = 1500, micro_batch_wait_ms: int = 5):
        self.tts = tts
        self.max_steps = int(max_steps)
        self.micro_batch_wait_s = float(micro_batch_wait_ms) / 1000.0
        self.prepare_coordinator = PrepareCoordinator(tts)
        self.condition = threading.Condition()
        self.prepare_inflight = 0
        self.prepare_peak_inflight = 0
        self.finalize_condition = threading.Condition()
        self.finalize_pending_tasks: Deque[SchedulerFinalizeTask] = deque()
        self.finalize_pending_peak = 0
        self.finalize_inflight = 0
        self.finalize_inflight_peak = 0
        self.finalize_workers = max(1, int(os.environ.get("GPTSOVITS_FINALIZE_WORKERS", 1)))
        self.finalize_mode = os.environ.get("GPTSOVITS_FINALIZE_MODE", "async").strip().lower()
        self.finalize_batch_max_items = max(1, int(os.environ.get("GPTSOVITS_FINALIZE_BATCH_MAX_ITEMS", 16)))
        self.finalize_batch_wait_s = max(0.0, float(os.environ.get("GPTSOVITS_FINALIZE_BATCH_WAIT_MS", "2")) / 1000.0)
        self.pending_jobs: List[SchedulerPendingJob] = []
        self.active_batch: T2SActiveBatch | None = None
        self.job_map: Dict[str, SchedulerPendingJob] = {}
        self.total_finished = 0
        self.total_submitted = 0
        self.worker_thread = threading.Thread(target=self._run_loop, name="unified-t2s-scheduler-worker", daemon=True)
        self.worker_thread.start()
        self.finalize_threads = [
            threading.Thread(
                target=self._run_finalize_loop,
                name=f"unified-t2s-finalize-{worker_index}",
                daemon=True,
            )
            for worker_index in range(self.finalize_workers)
        ]
        for finalize_thread in self.finalize_threads:
            finalize_thread.start()

    def snapshot(self) -> dict:
        with self.condition:
            finalize_pending = len(self.finalize_pending_tasks)
            prepare_state = self.prepare_coordinator.snapshot()
            return {
                "pending_jobs": len(self.pending_jobs),
                "running_requests": 0 if self.active_batch is None else len(self.active_batch.request_ids),
                "prepare_inflight": prepare_state["inflight"],
                "prepare_peak_inflight": prepare_state["peak_inflight"],
                "prepare_max_inflight": prepare_state.get("max_inflight", 0),
                "finalize_pending": finalize_pending,
                "finalize_pending_peak": self.finalize_pending_peak,
                "finalize_inflight": self.finalize_inflight,
                "finalize_inflight_peak": self.finalize_inflight_peak,
                "finalize_workers": self.finalize_workers,
                "finalize_mode": self.finalize_mode,
                "finalize_batch_max_items": self.finalize_batch_max_items,
                "finalize_batch_wait_ms": self.finalize_batch_wait_s * 1000.0,
                "total_submitted": self.total_submitted,
                "total_finished": self.total_finished,
                "drained": self.is_drained(),
            }

    def is_drained(self) -> bool:
        with self.condition:
            with self.finalize_condition:
                return (
                    self.active_batch is None
                    and not self.pending_jobs
                    and not self.job_map
                    and self.prepare_coordinator.snapshot()["inflight"] <= 0
                    and self.finalize_inflight <= 0
                    and not self.finalize_pending_tasks
                )

    def wait_until_idle(self, timeout_sec: float = 60.0, poll_interval_sec: float = 0.01) -> bool:
        deadline = time.perf_counter() + max(0.0, timeout_sec)
        while time.perf_counter() < deadline:
            if self.is_drained():
                return True
            time.sleep(poll_interval_sec)
        return self.is_drained()

    def _sync_device(self) -> None:
        try:
            device_str = str(self.tts.configs.device)
            if device_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(self.tts.configs.device)
            elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass

    def submit(
        self,
        state: T2SRequestState,
        speed_factor: float,
        sample_steps: int,
        media_type: str,
        prepare_wall_ms: float,
        prepare_profile_total_ms: float,
        done_loop: asyncio.AbstractEventLoop | None = None,
        done_future: asyncio.Future | None = None,
    ) -> SchedulerPendingJob:
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
            prepare_wall_ms=float(prepare_wall_ms),
            prepare_profile_total_ms=float(prepare_profile_total_ms),
        )
        with self.condition:
            self.pending_jobs.append(job)
            self.job_map[job.request_id] = job
            self.total_submitted += 1
            self.condition.notify_all()
        with self.finalize_condition:
            self.finalize_condition.notify_all()
        return job

    async def prepare_state_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> tuple[T2SRequestState, float, float]:
        with self.condition:
            self.prepare_inflight += 1
            self.prepare_peak_inflight = max(self.prepare_peak_inflight, self.prepare_inflight)
        try:
            return await self.prepare_coordinator.prepare_state_profiled_async(spec, prepare_submit_at)
        finally:
            with self.condition:
                self.prepare_inflight = max(0, self.prepare_inflight - 1)
                self.condition.notify_all()
            with self.finalize_condition:
                self.finalize_condition.notify_all()

    async def prepare_states_batch_async(self, specs: List[SchedulerRequestSpec]) -> List[T2SRequestState]:
        results = await asyncio.gather(
            *[self.prepare_state_profiled_async(spec, time.perf_counter()) for spec in specs]
        )
        return [state for state, _, _ in results]

    def _mark_prefill_started(self, pending_jobs: List[SchedulerPendingJob], started_at: float) -> None:
        with self.condition:
            for job in pending_jobs:
                tracked_job = self.job_map.get(job.request_id)
                if tracked_job is None:
                    continue
                tracked_job.first_schedule_time = float(started_at)

    def _add_prefill_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        if not request_ids:
            return
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
                if job is not None:
                    job.prefill_ms += delta_ms

    def _add_merge_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        if not request_ids:
            return
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
                if job is not None:
                    job.merge_ms += delta_ms

    def _add_decode_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        if not request_ids:
            return
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
                if job is not None:
                    job.decode_ms += delta_ms
                    job.decode_steps += 1

    def _add_finalize_wait_ms(self, request_ids: List[str], delta_ms: float) -> None:
        if not request_ids:
            return
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
                if job is not None:
                    job.finalize_wait_ms += float(delta_ms)

    def _enqueue_finalize_finished(self, items: List[T2SFinishedItem]) -> None:
        if not items:
            return
        enqueued_at = time.perf_counter()
        with self.finalize_condition:
            for item in items:
                self.finalize_pending_tasks.append(
                    SchedulerFinalizeTask(request_id=item.request_id, item=item, enqueued_time=enqueued_at)
                )
            self.finalize_pending_peak = max(self.finalize_pending_peak, len(self.finalize_pending_tasks))
            self.finalize_condition.notify_all()

    def _take_finalize_task_batch(self) -> List[SchedulerFinalizeTask]:
        with self.finalize_condition:
            while not self.finalize_pending_tasks:
                self.finalize_condition.wait()
            selected_tasks = [self.finalize_pending_tasks.popleft()]
            if self.finalize_mode == "sync" or self.tts.configs.use_vocoder:
                self.finalize_inflight += len(selected_tasks)
                self.finalize_inflight_peak = max(self.finalize_inflight_peak, self.finalize_inflight)
                return selected_tasks
            batch_deadline = time.perf_counter() + self.finalize_batch_wait_s
            while len(selected_tasks) < self.finalize_batch_max_items:
                if not self.finalize_pending_tasks:
                    remaining = batch_deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    self.finalize_condition.wait(timeout=remaining)
                    continue
                first_task = selected_tasks[0]
                matched_index = None
                for index, task in enumerate(self.finalize_pending_tasks):
                    if abs(task.enqueued_time - first_task.enqueued_time) < 1.0:
                        matched_index = index
                        break
                if matched_index is not None:
                    selected_tasks.append(self.finalize_pending_tasks[matched_index])
                    del self.finalize_pending_tasks[matched_index]
                    continue
                remaining = batch_deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.finalize_condition.wait(timeout=remaining)
            self.finalize_inflight += len(selected_tasks)
            self.finalize_inflight_peak = max(self.finalize_inflight_peak, self.finalize_inflight)
            return selected_tasks

    def _finalize_task_done(self, count: int) -> None:
        with self.finalize_condition:
            self.finalize_inflight = max(0, self.finalize_inflight - count)
            self.finalize_condition.notify_all()

    def _synthesize_finished_audio(self, job: SchedulerPendingJob, item: T2SFinishedItem) -> tuple[int, np.ndarray]:
        audio_fragment = self.tts.synthesize_audio_request_local(
            semantic_tokens=item.semantic_tokens.detach().clone().unsqueeze(0).unsqueeze(0),
            phones=job.state.phones.detach().clone().unsqueeze(0),
            prompt_semantic=job.state.prompt_semantic.detach().clone(),
            prompt_phones=job.state.prompt_phones.detach().clone(),
            refer_spec=(
                job.state.refer_spec[0].detach().clone(),
                None if job.state.refer_spec[1] is None else job.state.refer_spec[1].detach().clone(),
            ),
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
            super_sampling=False,
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
            refer_specs.append(
                (
                    job.state.refer_spec[0].detach().clone(),
                    None if job.state.refer_spec[1] is None else job.state.refer_spec[1].detach().clone(),
                )
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
                    super_sampling=False,
                )
            )
        return results

    def _complete_finalize_task(self, job: SchedulerPendingJob, item: T2SFinishedItem, sample_rate: int, audio_data: np.ndarray) -> None:
        finished_at = time.perf_counter()
        with self.condition:
            if self.job_map.get(item.request_id) is not job:
                return
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
            job.result = {
                "request_id": item.request_id,
                "semantic_len": int(item.semantic_tokens.shape[0]),
                "finish_idx": int(item.finish_idx),
                "finish_reason": item.finish_reason,
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
            job.done_event.set()
            self._notify_done_future(job)
            self.job_map.pop(item.request_id, None)
            self.total_finished += 1
            self.condition.notify_all()

    def _finalize_error(self, request_ids: List[str], error: str) -> None:
        if not request_ids:
            return
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
                if job is None:
                    continue
                job.error = error
                job.done_event.set()
                self._notify_done_future(job)
                self.job_map.pop(request_id, None)
                self.total_finished += 1
            self.condition.notify_all()

    @staticmethod
    def _resolve_done_future(job: SchedulerPendingJob) -> None:
        future = job.done_future
        if future is None or future.done():
            return
        future.set_result(True)

    def _notify_done_future(self, job: SchedulerPendingJob) -> None:
        if job.done_loop is None or job.done_future is None:
            return
        try:
            job.done_loop.call_soon_threadsafe(self._resolve_done_future, job)
        except RuntimeError:
            pass

    def _take_pending_snapshot(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
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

    def _run_finalize_loop(self) -> None:
        while True:
            tasks = self._take_finalize_task_batch()
            try:
                jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]] = []
                with self.condition:
                    for task in tasks:
                        job = self.job_map.get(task.request_id)
                        if job is None:
                            continue
                        jobs_and_items.append((job, task.item))
                if not jobs_and_items:
                    continue
                now = time.perf_counter()
                for task in tasks:
                    self._add_finalize_wait_ms([task.request_id], max(0.0, (now - task.enqueued_time) * 1000.0))
                self._sync_device()
                synth_start = time.perf_counter()
                if len(jobs_and_items) == 1 or self.tts.configs.use_vocoder:
                    job, item = jobs_and_items[0]
                    batch_results = [self._synthesize_finished_audio(job, item)]
                else:
                    batch_results = self._synthesize_finished_audio_batch(jobs_and_items)
                self._sync_device()
                synth_ms = (time.perf_counter() - synth_start) * 1000.0
                with self.condition:
                    for job, _ in jobs_and_items:
                        tracked_job = self.job_map.get(job.request_id)
                        if tracked_job is not None:
                            tracked_job.synth_ms += synth_ms
                for (job, item), (sample_rate, audio_data) in zip(jobs_and_items, batch_results):
                    self._complete_finalize_task(job, item, sample_rate=sample_rate, audio_data=audio_data)
            except Exception as exc:
                self._finalize_error([task.request_id for task in tasks], str(exc))
            finally:
                self._finalize_task_done(len(tasks))

    def _run_loop(self) -> None:
        while True:
            wait_for_batch = self.active_batch is None
            pending_jobs = self._take_pending_snapshot(wait_for_batch=wait_for_batch)

            if pending_jobs:
                try:
                    self._sync_device()
                    prefill_start = time.perf_counter()
                    self._mark_prefill_started(pending_jobs, prefill_start)
                    admitted_active_batch, admitted_finished = run_prefill_active_batch(
                        self.tts.t2s_model.model,
                        [job.state for job in pending_jobs],
                        max_steps=self.max_steps,
                    )
                    self._sync_device()
                    self._add_prefill_time([job.request_id for job in pending_jobs], time.perf_counter() - prefill_start)
                    self._enqueue_finalize_finished(admitted_finished)
                    merge_start = time.perf_counter()
                    self.active_batch = merge_active_batches(
                        self.tts.t2s_model.model,
                        self.active_batch,
                        admitted_active_batch,
                    )
                    self._add_merge_time(
                        [] if self.active_batch is None else list(self.active_batch.request_ids),
                        time.perf_counter() - merge_start,
                    )
                except Exception as exc:
                    self._finalize_error([job.request_id for job in pending_jobs], str(exc))

            if self.active_batch is not None:
                active_request_ids: List[str] = []
                try:
                    active_request_ids = [state.request_id for state in self.active_batch.states]
                    self._sync_device()
                    decode_start = time.perf_counter()
                    self.active_batch, step_finished = decode_one_step(
                        self.tts.t2s_model.model,
                        self.active_batch,
                        max_steps=self.max_steps,
                    )
                    self._sync_device()
                    self._add_decode_time(active_request_ids, time.perf_counter() - decode_start)
                    self._enqueue_finalize_finished(step_finished)
                except Exception as exc:
                    self._finalize_error(active_request_ids, str(exc))
                    self.active_batch = None
                continue

            if not pending_jobs:
                time.sleep(self.micro_batch_wait_s)


def set_scheduler_seed(seed: int):
    if seed in ["", None]:
        return
    seed = int(seed)
    if seed < 0:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    def handle_pack_ogg():
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)

    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except (RuntimeError, ValueError):
        handle_pack_ogg()
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-vn",
            "-f",
            "adts",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


class UnifiedTTSEngine:
    def __init__(
        self,
        tts: TTS,
        cut_method_names: Sequence[str],
        control_callbacks: RuntimeControlCallbacks | None = None,
        max_steps: int = 1500,
        micro_batch_wait_ms: int = 5,
    ) -> None:
        self.tts = tts
        self.cut_method_names = set(cut_method_names)
        self.control_callbacks = control_callbacks or RuntimeControlCallbacks()
        self.reference_registry = ReferenceRegistry()
        self.model_registry = ModelRegistry(
            t2s_weights_path=str(self.tts.configs.t2s_weights_path),
            vits_weights_path=str(self.tts.configs.vits_weights_path),
        )
        self.scheduler_worker = UnifiedSchedulerWorker(tts, max_steps=max_steps, micro_batch_wait_ms=micro_batch_wait_ms)
        self.direct_tts_lock = threading.RLock()
        self.management_lock = threading.RLock()

    def _normalize_lang(self, value: str | None) -> str | None:
        if value in [None, ""]:
            return value
        return str(value).lower()

    def _apply_default_reference(self, req: dict) -> dict:
        normalized = dict(req)
        default_ref = self.reference_registry.get_default()
        if normalized.get("ref_audio_path") in [None, ""] and default_ref.ref_audio_path not in [None, ""]:
            normalized["ref_audio_path"] = default_ref.ref_audio_path
        if "text_lang" in normalized:
            normalized["text_lang"] = self._normalize_lang(normalized.get("text_lang"))
        if "prompt_lang" in normalized:
            normalized["prompt_lang"] = self._normalize_lang(normalized.get("prompt_lang"))
        return normalized

    def check_params(self, req: dict) -> Optional[str]:
        text = req.get("text", "")
        text_lang = req.get("text_lang", "")
        ref_audio_path = req.get("ref_audio_path", "")
        media_type = req.get("media_type", "wav")
        prompt_lang = req.get("prompt_lang", "")
        text_split_method = req.get("text_split_method", "cut5")

        if ref_audio_path in [None, ""]:
            return "ref_audio_path is required"
        if text in [None, ""]:
            return "text is required"
        if text_lang in [None, ""]:
            return "text_lang is required"
        if text_lang.lower() not in self.tts.configs.languages:
            return f"text_lang: {text_lang} is not supported in version {self.tts.configs.version}"
        if prompt_lang in [None, ""]:
            return "prompt_lang is required"
        if prompt_lang.lower() not in self.tts.configs.languages:
            return f"prompt_lang: {prompt_lang} is not supported in version {self.tts.configs.version}"
        if media_type not in ["wav", "raw", "ogg", "aac"]:
            return f"media_type: {media_type} is not supported"
        if text_split_method not in self.cut_method_names:
            return f"text_split_method:{text_split_method} is not supported"
        return None

    @staticmethod
    def _normalize_streaming_mode(req: dict) -> dict:
        normalized = dict(req)
        streaming_mode = normalized.get("streaming_mode", False)
        return_fragment = normalized.get("return_fragment", False)
        if streaming_mode is False:
            normalized["streaming_mode"] = False
            normalized["return_fragment"] = False
            normalized["fixed_length_chunk"] = False
        elif streaming_mode == 0:
            normalized["streaming_mode"] = False
            normalized["return_fragment"] = False
            normalized["fixed_length_chunk"] = False
        elif streaming_mode == 1 or streaming_mode is True:
            normalized["streaming_mode"] = False
            normalized["return_fragment"] = True
            normalized["fixed_length_chunk"] = False
        elif streaming_mode == 2:
            normalized["streaming_mode"] = True
            normalized["return_fragment"] = False
            normalized["fixed_length_chunk"] = False
        elif streaming_mode == 3:
            normalized["streaming_mode"] = True
            normalized["return_fragment"] = False
            normalized["fixed_length_chunk"] = True
        else:
            raise ValueError("the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)")
        normalized["response_streaming"] = bool(normalized["streaming_mode"] or normalized["return_fragment"] or return_fragment)
        return normalized

    def _iter_direct_tts_bytes(self, req: dict) -> Generator[bytes, None, None]:
        media_type = req["media_type"]
        with self.direct_tts_lock:
            tts_generator = self.tts.run(req)
            first_chunk = True
            current_media_type = media_type
            for sr, chunk in tts_generator:
                if first_chunk and media_type == "wav":
                    yield wave_header_chunk(sample_rate=sr)
                    current_media_type = "raw"
                    first_chunk = False
                yield pack_audio(BytesIO(), chunk, sr, current_media_type).getvalue()

    def run_direct_tts(self, req: dict) -> DirectTTSExecution:
        normalized = self._normalize_streaming_mode(self._apply_default_reference(req))
        error = self.check_params(normalized)
        if error is not None:
            raise ValueError(error)
        media_type = normalized.get("media_type", "wav")
        if normalized["response_streaming"]:
            return DirectTTSExecution(
                media_type=media_type,
                streaming=True,
                audio_generator=self._iter_direct_tts_bytes(normalized),
            )
        with self.direct_tts_lock:
            tts_generator = self.tts.run(normalized)
            sr, audio_data = next(tts_generator)
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=pack_audio(BytesIO(), audio_data, sr, media_type).getvalue(),
        )

    def build_scheduler_request_specs(self, request_items: List[dict]) -> List[SchedulerRequestSpec]:
        specs: List[SchedulerRequestSpec] = []
        for index, payload in enumerate(request_items):
            req = self._apply_default_reference(
                {
                    "text": payload["text"],
                    "text_lang": self._normalize_lang(payload["text_lang"]),
                    "ref_audio_path": payload["ref_audio_path"],
                    "aux_ref_audio_paths": None,
                    "prompt_text": payload["prompt_text"],
                    "prompt_lang": self._normalize_lang(payload["prompt_lang"]),
                    "top_k": payload["top_k"],
                    "top_p": payload["top_p"],
                    "temperature": payload["temperature"],
                    "text_split_method": "cut5",
                    "batch_size": 1,
                    "batch_threshold": 0.75,
                    "speed_factor": 1.0,
                    "split_bucket": False,
                    "fragment_interval": 0.3,
                    "seed": -1,
                    "media_type": "wav",
                    "streaming_mode": False,
                    "parallel_infer": False,
                    "repetition_penalty": payload["repetition_penalty"],
                    "sample_steps": 32,
                    "super_sampling": False,
                    "overlap_length": 2,
                    "min_chunk_length": 16,
                }
            )
            error = self.check_params(req)
            if error is not None:
                raise ValueError(f"request[{index}] 参数非法: {error}")
            specs.append(
                SchedulerRequestSpec(
                    request_id=payload.get("request_id") or f"req_{index:03d}",
                    ref_audio_path=Path(req["ref_audio_path"]),
                    prompt_text=payload["prompt_text"],
                    prompt_lang=req["prompt_lang"],
                    text=payload["text"],
                    text_lang=req["text_lang"],
                    top_k=int(payload["top_k"]),
                    top_p=float(payload["top_p"]),
                    temperature=float(payload["temperature"]),
                    repetition_penalty=float(payload["repetition_penalty"]),
                    early_stop_num=int(payload.get("early_stop_num", -1)),
                    ready_step=int(payload.get("ready_step", 0)),
                )
            )
        return specs

    def build_scheduler_submit_spec(self, payload: dict) -> SchedulerRequestSpec:
        request_id = payload.get("request_id") or f"job_{uuid.uuid4().hex[:12]}"
        req = self._apply_default_reference(
            {
                "text": payload["text"],
                "text_lang": self._normalize_lang(payload["text_lang"]),
                "ref_audio_path": payload["ref_audio_path"],
                "aux_ref_audio_paths": None,
                "prompt_text": payload["prompt_text"],
                "prompt_lang": self._normalize_lang(payload["prompt_lang"]),
                "top_k": payload["top_k"],
                "top_p": payload["top_p"],
                "temperature": payload["temperature"],
                "text_split_method": "cut5",
                "batch_size": 1,
                "batch_threshold": 0.75,
                "speed_factor": float(payload["speed_factor"]),
                "split_bucket": False,
                "fragment_interval": 0.3,
                "seed": -1,
                "media_type": payload["media_type"],
                "streaming_mode": False,
                "parallel_infer": False,
                "repetition_penalty": payload["repetition_penalty"],
                "sample_steps": int(payload["sample_steps"]),
                "super_sampling": False,
                "overlap_length": 2,
                "min_chunk_length": 16,
            }
        )
        error = self.check_params(req)
        if error is not None:
            raise ValueError(f"request 参数非法: {error}")
        return SchedulerRequestSpec(
            request_id=request_id,
            ref_audio_path=Path(req["ref_audio_path"]),
            prompt_text=payload["prompt_text"],
            prompt_lang=req["prompt_lang"],
            text=payload["text"],
            text_lang=req["text_lang"],
            top_k=int(payload["top_k"]),
            top_p=float(payload["top_p"]),
            temperature=float(payload["temperature"]),
            repetition_penalty=float(payload["repetition_penalty"]),
            early_stop_num=int(payload.get("early_stop_num", -1)),
            ready_step=0,
        )

    @staticmethod
    def summarize_scheduler_states(states: List[T2SRequestState]) -> List[dict]:
        return [
            {
                "request_id": state.request_id,
                "ready_step": int(state.ready_step),
                "ref_audio_path": str(state.ref_audio_path),
                "prompt_semantic_len": int(state.prompt_semantic.shape[0]),
                "all_phone_len": int(state.all_phones.shape[0]),
                "bert_len": int(state.all_bert_features.shape[-1]),
                "norm_text": state.norm_text,
            }
            for state in states
        ]

    @staticmethod
    def summarize_scheduler_finished(items: List[T2SFinishedItem]) -> List[dict]:
        return [
            {
                "request_id": item.request_id,
                "semantic_len": int(item.semantic_tokens.shape[0]),
                "finish_idx": int(item.finish_idx),
                "finish_reason": item.finish_reason,
            }
            for item in items
        ]

    async def run_scheduler_debug(self, request_items: List[dict], max_steps: int, seed: int) -> SchedulerDebugExecution:
        set_scheduler_seed(seed)
        specs = self.build_scheduler_request_specs(request_items)
        states = await self.scheduler_worker.prepare_states_batch_async(specs)
        finished = run_scheduler_continuous(self.tts.t2s_model.model, states, max_steps=int(max_steps))
        return SchedulerDebugExecution(
            payload={
                "message": "success",
                "request_count": len(states),
                "max_steps": int(max_steps),
                "requests": self.summarize_scheduler_states(states),
                "finished": self.summarize_scheduler_finished(finished),
            }
        )

    async def run_scheduler_submit(self, payload: dict) -> SchedulerSubmitExecution:
        request_start = time.perf_counter()
        prepare_start = request_start
        spec = self.build_scheduler_submit_spec(payload)
        spec_ready_at = time.perf_counter()
        prepare_spec_build_ms = max(0.0, (spec_ready_at - prepare_start) * 1000.0)
        state, prepare_exec_started_at, prepare_exec_finished_at = await self.scheduler_worker.prepare_state_profiled_async(
            spec,
            spec_ready_at,
        )
        prepare_wall_ms = max(0.0, (prepare_exec_finished_at - spec_ready_at) * 1000.0)
        prepare_executor_queue_ms = max(0.0, (prepare_exec_started_at - spec_ready_at) * 1000.0)
        prepare_executor_run_ms = max(0.0, (prepare_exec_finished_at - prepare_exec_started_at) * 1000.0)
        prepare_profile = dict(state.prepare_profile)
        prepare_profile_total_ms = float(prepare_profile.get("wall_total_ms", prepare_wall_ms))
        prepare_profile_wall_ms = float(prepare_profile.get("wall_total_ms", prepare_wall_ms))
        prepare_other_ms = max(0.0, prepare_wall_ms - prepare_spec_build_ms - prepare_executor_queue_ms - prepare_executor_run_ms)
        api_after_prepare_start = time.perf_counter()
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        job = self.scheduler_worker.submit(
            state=state,
            speed_factor=float(payload["speed_factor"]),
            sample_steps=int(payload["sample_steps"]),
            media_type=str(payload["media_type"]),
            prepare_wall_ms=prepare_wall_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            done_loop=loop,
            done_future=done_future,
        )
        api_after_prepare_ms = max(0.0, (time.perf_counter() - api_after_prepare_start) * 1000.0)
        await asyncio.wait_for(done_future, timeout=float(payload.get("timeout_sec", 30.0)))
        wait_return_at = time.perf_counter()
        if job.error is not None:
            raise RuntimeError(job.error)
        if job.audio_data is None or job.sample_rate is None or job.result is None:
            raise RuntimeError(f"{job.request_id} finished without audio result")
        pack_start = time.perf_counter()
        audio_data = pack_audio(BytesIO(), job.audio_data, int(job.sample_rate), job.media_type).getvalue()
        pack_end = time.perf_counter()
        pack_ms = (pack_end - pack_start) * 1000.0
        api_wait_result_ms = 0.0
        if job.result_ready_time is not None:
            api_wait_result_ms = max(0.0, (wait_return_at - job.result_ready_time) * 1000.0)
        worker_total_ms = float(job.result["worker_total_ms"]) if job.result is not None else 0.0
        headers = {
            "X-Request-Id": job.request_id,
            "X-Semantic-Len": str(job.result["semantic_len"]) if job.result is not None else "0",
            "X-Finish-Reason": job.result["finish_reason"] if job.result is not None else "unknown",
            "X-Queue-Wait-Ms": f"{float(job.result['queue_wait_ms']):.3f}" if job.result is not None else "0.000",
            "X-Prepare-Ms": f"{prepare_wall_ms:.3f}",
            "X-Prepare-Wall-Ms": f"{prepare_wall_ms:.3f}",
            "X-Prepare-Spec-Build-Ms": f"{prepare_spec_build_ms:.3f}",
            "X-Prepare-Executor-Queue-Ms": f"{prepare_executor_queue_ms:.3f}",
            "X-Prepare-Admission-Wait-Ms": (
                f"{float(job.result['prepare_profile'].get('prepare_admission_wait_ms', 0.0)):.3f}"
                if job.result is not None
                else "0.000"
            ),
            "X-Prepare-Executor-Run-Ms": f"{prepare_executor_run_ms:.3f}",
            "X-Prepare-Profile-Total-Ms": f"{prepare_profile_total_ms:.3f}",
            "X-Prepare-Profile-Wall-Ms": f"{prepare_profile_wall_ms:.3f}",
            "X-Prepare-Other-Ms": f"{prepare_other_ms:.3f}",
            "X-Api-After-Prepare-Ms": f"{api_after_prepare_ms:.3f}",
            "X-Prefill-Ms": f"{float(job.result['prefill_ms']):.3f}" if job.result is not None else "0.000",
            "X-Merge-Ms": f"{float(job.result['merge_ms']):.3f}" if job.result is not None else "0.000",
            "X-Decode-Ms": f"{float(job.result['decode_ms']):.3f}" if job.result is not None else "0.000",
            "X-Finalize-Wait-Ms": f"{float(job.result['finalize_wait_ms']):.3f}" if job.result is not None else "0.000",
            "X-Synth-Ms": f"{float(job.result['synth_ms']):.3f}" if job.result is not None else "0.000",
            "X-Worker-Residual-Ms": f"{float(job.result['worker_residual_ms']):.3f}" if job.result is not None else "0.000",
            "X-Worker-Other-Ms": f"{float(job.result['worker_other_ms']):.3f}" if job.result is not None else "0.000",
            "X-Pack-Ms": f"{pack_ms:.3f}",
            "X-Worker-Total-Ms": f"{float(job.result['worker_total_ms']):.3f}" if job.result is not None else "0.000",
            "X-Api-Wait-Result-Ms": f"{api_wait_result_ms:.3f}",
            "X-Decode-Steps": str(int(job.result["decode_steps"])) if job.result is not None else "0",
            "X-Sample-Rate": str(int(job.sample_rate)),
        }
        prepare_profile = job.result.get("prepare_profile", {}) if job.result is not None else {}
        if job.result is not None:
            headers.update(
                {
                    "X-Prepare-Prompt-Text-Ms": f"{float(prepare_profile.get('prompt_text_features_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Text-Ms": f"{float(prepare_profile.get('text_features_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Text-CPU-Preprocess-Ms": f"{float(prepare_profile.get('prompt_text_cpu_preprocess_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Text-CPU-Preprocess-Ms": f"{float(prepare_profile.get('text_cpu_preprocess_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Text-CPU-Queue-Ms": f"{float(prepare_profile.get('prompt_text_cpu_queue_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Text-CPU-Queue-Ms": f"{float(prepare_profile.get('text_cpu_queue_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Text-Feature-Queue-Ms": f"{float(prepare_profile.get('prompt_text_feature_queue_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Text-Feature-Queue-Ms": f"{float(prepare_profile.get('text_feature_queue_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Bert-Wait-Ms": f"{float(prepare_profile.get('prompt_text_bert_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Bert-Wait-Ms": f"{float(prepare_profile.get('text_bert_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Bert-Admission-Wait-Ms": f"{float(prepare_profile.get('prompt_text_bert_admission_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Bert-Admission-Wait-Ms": f"{float(prepare_profile.get('text_bert_admission_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Bert-Queue-Wait-Ms": f"{float(prepare_profile.get('prompt_text_bert_queue_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Bert-Queue-Wait-Ms": f"{float(prepare_profile.get('text_bert_queue_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Bert-Batch-Collect-Wait-Ms": f"{float(prepare_profile.get('prompt_text_bert_batch_collect_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Bert-Batch-Collect-Wait-Ms": f"{float(prepare_profile.get('text_bert_batch_collect_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Bert-Forward-Ms": f"{float(prepare_profile.get('prompt_text_bert_forward_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Bert-Forward-Ms": f"{float(prepare_profile.get('text_bert_forward_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Bert-Pending-On-Enqueue-Peak": str(int(prepare_profile.get("prompt_text_bert_pending_depth_on_enqueue_peak", 0.0))),
                    "X-Prepare-Target-Bert-Pending-On-Enqueue-Peak": str(int(prepare_profile.get("text_bert_pending_depth_on_enqueue_peak", 0.0))),
                    "X-Prepare-Prompt-Bert-Pending-On-Collect-Peak": str(int(prepare_profile.get("prompt_text_bert_pending_depth_on_collect_peak", 0.0))),
                    "X-Prepare-Target-Bert-Pending-On-Collect-Peak": str(int(prepare_profile.get("text_bert_pending_depth_on_collect_peak", 0.0))),
                    "X-Prepare-Prompt-Bert-High-Pressure-Peak": str(int(prepare_profile.get("prompt_text_bert_high_pressure_mode_peak", 0.0))),
                    "X-Prepare-Target-Bert-High-Pressure-Peak": str(int(prepare_profile.get("text_bert_high_pressure_mode_peak", 0.0))),
                    "X-Prepare-Prompt-Bert-Batch-Window-Ms": f"{float(prepare_profile.get('prompt_text_bert_batch_window_ms', 0.0)):.3f}",
                    "X-Prepare-Target-Bert-Batch-Window-Ms": f"{float(prepare_profile.get('text_bert_batch_window_ms', 0.0)):.3f}",
                    "X-Prepare-Text-Pair-Wall-Ms": f"{float(prepare_profile.get('text_feature_pair_ms', 0.0)):.3f}",
                    "X-Prepare-Text-CPU-Workers": str(int(prepare_profile.get('text_cpu_parallel_workers', 0.0))),
                    "X-Prepare-Audio-Load-Ms": f"{float(prepare_profile.get('audio_load_ms', 0.0)):.3f}",
                    "X-Prepare-Audio-Stage-Wait-Ms": f"{float(prepare_profile.get('audio_stage_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Semantic-Ms": f"{float(prepare_profile.get('prompt_semantic_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Semantic-Wait-Ms": f"{float(prepare_profile.get('prompt_semantic_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Semantic-CPU-Ms": f"{float(prepare_profile.get('prompt_semantic_cpu_prepare_ms', 0.0)):.3f}",
                    "X-Prepare-Prompt-Semantic-Forward-Ms": f"{float(prepare_profile.get('prompt_semantic_forward_ms', 0.0)):.3f}",
                    "X-Prepare-Ref-Spec-Ms": f"{float(prepare_profile.get('ref_spec_ms', 0.0)):.3f}",
                    "X-Prepare-Ref-Spec-Wait-Ms": f"{float(prepare_profile.get('ref_spec_wait_ms', 0.0)):.3f}",
                    "X-Prepare-Ref-Bundle-Ms": f"{float(prepare_profile.get('ref_audio_bundle_ms', 0.0)):.3f}",
                    "X-Prepare-Tensorize-Ms": f"{float(prepare_profile.get('tensorize_ms', 0.0)):.3f}",
                    "X-Prepare-Inflight-On-Enter": str(int(prepare_profile.get('worker_prepare_inflight_on_enter', 0.0))),
                    "X-Prepare-Inflight-Peak": str(int(prepare_profile.get('worker_prepare_peak_inflight', 0.0))),
                }
            )
        response_ready_at = time.perf_counter()
        response_overhead_ms = max(0.0, (response_ready_at - pack_end) * 1000.0)
        request_total_ms = max(0.0, (response_ready_at - request_start) * 1000.0)
        request_other_ms = max(
            0.0,
            request_total_ms - prepare_wall_ms - api_after_prepare_ms - worker_total_ms - api_wait_result_ms - pack_ms,
        )
        headers["X-Response-Overhead-Ms"] = f"{response_overhead_ms:.3f}"
        headers["X-Request-Other-Ms"] = f"{request_other_ms:.3f}"
        headers["X-Request-Total-Ms"] = f"{request_total_ms:.3f}"
        return SchedulerSubmitExecution(audio_bytes=audio_data, media_type=f"audio/{job.media_type}", headers=headers)

    def get_scheduler_state(self) -> dict:
        return self.scheduler_worker.snapshot()

    def get_runtime_state(self) -> dict:
        model_state = self.model_registry.snapshot()
        default_ref = self.reference_registry.get_default()
        scheduler_state = self.get_scheduler_state()
        return {
            "message": "success",
            "default_reference": {
                "ref_audio_path": default_ref.ref_audio_path,
                "updated_at": default_ref.updated_at,
            },
            "model_registry": {
                "generation": model_state.generation,
                "t2s_generation": model_state.t2s_generation,
                "vits_generation": model_state.vits_generation,
                "t2s_weights_path": model_state.t2s_weights_path,
                "vits_weights_path": model_state.vits_weights_path,
                "updated_at": model_state.updated_at,
            },
            "worker_state": scheduler_state,
        }

    def _wait_for_safe_reload(self, timeout_sec: float = 300.0) -> None:
        if not self.scheduler_worker.wait_until_idle(timeout_sec=timeout_sec):
            raise TimeoutError("scheduler worker did not drain before model reload")

    def set_refer_audio(self, refer_audio_path: str | None) -> dict:
        if refer_audio_path in [None, ""]:
            state = self.reference_registry.clear()
            return {"message": "success", "default_ref_audio_path": state.ref_audio_path}
        if not os.path.exists(str(refer_audio_path)):
            raise FileNotFoundError(f"{refer_audio_path} not exists")
        with self.management_lock:
            with self.direct_tts_lock:
                self.tts.set_ref_audio(str(refer_audio_path))
            state = self.reference_registry.set_default(str(refer_audio_path))
        return {"message": "success", "default_ref_audio_path": state.ref_audio_path}

    def set_gpt_weights(self, weights_path: str) -> dict:
        if weights_path in ["", None]:
            raise ValueError("gpt weight path is required")
        with self.management_lock:
            self._wait_for_safe_reload()
            with self.direct_tts_lock:
                self.tts.init_t2s_weights(weights_path)
                self.tts.refresh_runtime_components()
            state = self.model_registry.mark_t2s_reload(str(weights_path))
        return {"message": "success", "t2s_generation": state.t2s_generation, "generation": state.generation}

    def set_sovits_weights(self, weights_path: str) -> dict:
        if weights_path in ["", None]:
            raise ValueError("sovits weight path is required")
        with self.management_lock:
            self._wait_for_safe_reload()
            with self.direct_tts_lock:
                self.tts.init_vits_weights(weights_path)
                self.tts.refresh_runtime_components()
            state = self.model_registry.mark_vits_reload(str(weights_path))
        return {"message": "success", "vits_generation": state.vits_generation, "generation": state.generation}

    def handle_control(self, command: str) -> None:
        if command == "restart":
            if self.control_callbacks.restart is None:
                os.execl(sys.executable, sys.executable, *sys.argv)
            self.control_callbacks.restart()
            return
        if command == "exit":
            if self.control_callbacks.exit is None:
                os.kill(os.getpid(), signal.SIGTERM)
                return
            self.control_callbacks.exit()
            return
        raise ValueError(f"unsupported command: {command}")
