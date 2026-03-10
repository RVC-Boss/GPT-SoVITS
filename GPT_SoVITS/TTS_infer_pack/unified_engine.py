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
    request_id: Optional[str] = None


@dataclass
class NormalizedEngineRequest:
    request_id: str
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_lang: str
    prompt_text: str = ""
    aux_ref_audio_paths: List[str] | None = None
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    early_stop_num: int = -1
    ready_step: int = 0
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = False
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool | int = False
    return_fragment: bool = False
    fixed_length_chunk: bool = False
    response_streaming: bool = False
    parallel_infer: bool = False
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    timeout_sec: float | None = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text": self.text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "aux_ref_audio_paths": list(self.aux_ref_audio_paths) if self.aux_ref_audio_paths else None,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "batch_threshold": self.batch_threshold,
            "speed_factor": self.speed_factor,
            "split_bucket": self.split_bucket,
            "fragment_interval": self.fragment_interval,
            "seed": self.seed,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,
            "return_fragment": self.return_fragment,
            "fixed_length_chunk": self.fixed_length_chunk,
            "response_streaming": self.response_streaming,
            "parallel_infer": self.parallel_infer,
            "repetition_penalty": self.repetition_penalty,
            "sample_steps": self.sample_steps,
            "super_sampling": self.super_sampling,
            "overlap_length": self.overlap_length,
            "min_chunk_length": self.min_chunk_length,
            "early_stop_num": self.early_stop_num,
            "ready_step": self.ready_step,
            "timeout_sec": self.timeout_sec,
        }

    def to_scheduler_spec(self) -> SchedulerRequestSpec:
        return SchedulerRequestSpec(
            request_id=self.request_id,
            ref_audio_path=Path(self.ref_audio_path),
            prompt_text=self.prompt_text,
            prompt_lang=self.prompt_lang,
            text=self.text,
            text_lang=self.text_lang,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            early_stop_num=self.early_stop_num,
            ready_step=self.ready_step,
        )


@dataclass
class SchedulerDebugExecution:
    payload: Dict[str, Any]


@dataclass
class SchedulerSubmitExecution:
    audio_bytes: bytes
    media_type: str
    headers: Dict[str, str]


class EngineStatus:
    NEW = "NEW"
    QUEUED = "QUEUED"
    VALIDATED = "VALIDATED"
    CPU_PREPARING = "CPU_PREPARING"
    GPU_PREPARING = "GPU_PREPARING"
    READY_FOR_PREFILL = "READY_FOR_PREFILL"
    ACTIVE_DECODE = "ACTIVE_DECODE"
    READY_FOR_FINALIZE = "READY_FOR_FINALIZE"
    FINALIZING = "FINALIZING"
    STREAMING = "STREAMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class EngineRequestState:
    request_id: str
    api_mode: str
    backend: str
    media_type: str
    response_streaming: bool
    submit_ts: float
    deadline_ts: float | None = None
    status: str = EngineStatus.NEW
    updated_ts: float = 0.0
    error: str | None = None
    finish_reason: str | None = None
    meta: Dict[str, Any] = field(default_factory=dict)
    profile: Dict[str, Any] = field(default_factory=dict)
    lifecycle_timestamps: Dict[str, float] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "api_mode": self.api_mode,
            "backend": self.backend,
            "media_type": self.media_type,
            "response_streaming": self.response_streaming,
            "status": self.status,
            "submit_ts": self.submit_ts,
            "updated_ts": self.updated_ts,
            "deadline_ts": self.deadline_ts,
            "error": self.error,
            "finish_reason": self.finish_reason,
            "meta": dict(self.meta),
            "profile": dict(self.profile),
            "lifecycle_timestamps": dict(self.lifecycle_timestamps),
        }


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
    admission_wait_ms: float = 0.0
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
    engine_request_id: str | None = None


@dataclass
class SchedulerFinalizeTask:
    request_id: str
    item: T2SFinishedItem
    enqueued_time: float


@dataclass
class RuntimeStateCallbacks:
    update: Callable[[str, str, Optional[Dict[str, Any]]], None] | None = None
    complete: Callable[[str, Optional[Dict[str, Any]]], None] | None = None
    fail: Callable[[str, str], None] | None = None


class UnifiedSchedulerWorker:
    def __init__(
        self,
        tts: TTS,
        max_steps: int = 1500,
        micro_batch_wait_ms: int = 5,
        runtime_callbacks: RuntimeStateCallbacks | None = None,
    ):
        self.tts = tts
        self.max_steps = int(max_steps)
        self.micro_batch_wait_s = float(micro_batch_wait_ms) / 1000.0
        self.runtime_callbacks = runtime_callbacks or RuntimeStateCallbacks()
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
        self.decode_backlog_max = max(0, int(os.environ.get("GPTSOVITS_ENGINE_DECODE_BACKLOG_MAX", "0")))
        self.finalize_pending_max = max(0, int(os.environ.get("GPTSOVITS_ENGINE_FINALIZE_PENDING_MAX", "0")))
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

    def _current_decode_backlog_locked(self) -> int:
        running_requests = 0 if self.active_batch is None else len(self.active_batch.request_ids)
        return int(len(self.pending_jobs) + running_requests)

    def _can_accept_submit_locked(self) -> tuple[bool, Dict[str, int]]:
        decode_backlog = self._current_decode_backlog_locked()
        finalize_pending = int(len(self.finalize_pending_tasks))
        prepare_inflight = int(self.prepare_coordinator.snapshot()["inflight"])
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
        last_snapshot: Dict[str, int] = {}
        while True:
            with self.condition:
                allowed, snapshot = self._can_accept_submit_locked()
                last_snapshot = snapshot
                if allowed:
                    return max(0.0, (time.perf_counter() - start) * 1000.0), snapshot
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError(
                        "scheduler submit admission timeout "
                        f"(decode_backlog={snapshot['decode_backlog']}, finalize_pending={snapshot['finalize_pending']})"
                    )
                self.condition.wait(timeout=self.micro_batch_wait_s)

    async def submit_async(
        self,
        state: T2SRequestState,
        speed_factor: float,
        sample_steps: int,
        media_type: str,
        prepare_wall_ms: float,
        prepare_profile_total_ms: float,
        done_loop: asyncio.AbstractEventLoop | None = None,
        done_future: asyncio.Future | None = None,
        engine_request_id: str | None = None,
        timeout_sec: float | None = None,
    ) -> SchedulerPendingJob:
        return await asyncio.to_thread(
            self.submit,
            state,
            speed_factor,
            sample_steps,
            media_type,
            prepare_wall_ms,
            prepare_profile_total_ms,
            done_loop,
            done_future,
            engine_request_id,
            timeout_sec,
        )

    def snapshot(self) -> dict:
        with self.condition:
            finalize_pending = len(self.finalize_pending_tasks)
            prepare_state = self.prepare_coordinator.snapshot()
            active_batch = self.active_batch
            active_batch_summary = None
            if active_batch is not None:
                active_batch_summary = {
                    "request_count": int(len(active_batch.request_ids)),
                    "request_ids": list(active_batch.request_ids),
                    "prefill_done": bool(active_batch.prefill_done),
                    "decode_step_index_max": (
                        int(active_batch.step_indices.max().item())
                        if active_batch.step_indices is not None and active_batch.step_indices.numel() > 0
                        else 0
                    ),
                }
            return {
                "pending_jobs": len(self.pending_jobs),
                "running_requests": 0 if active_batch is None else len(active_batch.request_ids),
                "prepare_inflight": prepare_state["inflight"],
                "prepare_peak_inflight": prepare_state["peak_inflight"],
                "prepare_max_inflight": prepare_state.get("max_inflight", 0),
                "prepare_state": dict(prepare_state),
                "finalize_pending": finalize_pending,
                "finalize_pending_peak": self.finalize_pending_peak,
                "finalize_inflight": self.finalize_inflight,
                "finalize_inflight_peak": self.finalize_inflight_peak,
                "finalize_workers": self.finalize_workers,
                "finalize_mode": self.finalize_mode,
                "finalize_batch_max_items": self.finalize_batch_max_items,
                "finalize_batch_wait_ms": self.finalize_batch_wait_s * 1000.0,
                "decode_backlog_max": self.decode_backlog_max,
                "finalize_pending_max": self.finalize_pending_max,
                "active_batch": active_batch_summary,
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
        engine_request_id: str | None = None,
        timeout_sec: float | None = None,
    ) -> SchedulerPendingJob:
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
            admission_wait_ms=float(admission_wait_ms),
            prepare_wall_ms=float(prepare_wall_ms),
            prepare_profile_total_ms=float(prepare_profile_total_ms),
            engine_request_id=engine_request_id or state.request_id,
        )
        with self.condition:
            self.pending_jobs.append(job)
            self.job_map[job.request_id] = job
            self.total_submitted += 1
            self.condition.notify_all()
        self._runtime_update(
            job.engine_request_id,
            EngineStatus.QUEUED,
            {
                "scheduler_request_id": job.request_id,
                "decode_admission_wait_ms": float(admission_wait_ms),
                "admission_snapshot": dict(admission_snapshot),
            },
        )
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
                self._runtime_update(
                    tracked_job.engine_request_id,
                    EngineStatus.GPU_PREPARING,
                    {"scheduler_request_id": tracked_job.request_id, "prefill_started_at": float(started_at)},
                )

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
        activate_request_ids: List[str] = []
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
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
                job = self.job_map.get(request_id)
                if job is not None:
                    job.finalize_wait_ms += float(delta_ms)

    def _enqueue_finalize_finished(self, items: List[T2SFinishedItem]) -> None:
        if not items:
            return
        enqueued_at = time.perf_counter()
        with self.finalize_condition:
            for item in items:
                job = self.job_map.get(item.request_id)
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
            with self.condition:
                self.condition.notify_all()
            return selected_tasks

    def _finalize_task_done(self, count: int) -> None:
        with self.finalize_condition:
            self.finalize_inflight = max(0, self.finalize_inflight - count)
            self.finalize_condition.notify_all()
        with self.condition:
            self.condition.notify_all()

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
                "decode_admission_wait_ms": float(job.admission_wait_ms),
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
        self._runtime_complete(
            job.engine_request_id,
            {
                "finish_reason": item.finish_reason,
                "semantic_len": int(item.semantic_tokens.shape[0]),
                "finish_idx": int(item.finish_idx),
                "sample_rate": int(sample_rate),
                "worker_profile": dict(job.result or {}),
            },
        )

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
                self._runtime_fail(job.engine_request_id, error)
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

    def _runtime_update(self, request_id: str | None, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if request_id is None or self.runtime_callbacks.update is None:
            return
        self.runtime_callbacks.update(request_id, status, extra)

    def _runtime_complete(self, request_id: str | None, extra: Optional[Dict[str, Any]] = None) -> None:
        if request_id is None or self.runtime_callbacks.complete is None:
            return
        self.runtime_callbacks.complete(request_id, extra)

    def _runtime_fail(self, request_id: str | None, error: str) -> None:
        if request_id is None or self.runtime_callbacks.fail is None:
            return
        self.runtime_callbacks.fail(request_id, error)

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
                for job, item in jobs_and_items:
                    self._runtime_update(
                        job.engine_request_id,
                        EngineStatus.FINALIZING,
                        {
                            "finish_reason": item.finish_reason,
                            "semantic_len": int(item.semantic_tokens.shape[0]),
                        },
                    )
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
        self.request_registry_lock = threading.Lock()
        self.active_requests: Dict[str, EngineRequestState] = {}
        self.recent_requests: Deque[EngineRequestState] = deque()
        self.recent_request_limit = max(1, int(os.environ.get("GPTSOVITS_ENGINE_RECENT_REQUEST_LIMIT", "64")))
        self.scheduler_worker = UnifiedSchedulerWorker(
            tts,
            max_steps=max_steps,
            micro_batch_wait_ms=micro_batch_wait_ms,
            runtime_callbacks=RuntimeStateCallbacks(
                update=self._update_request_state,
                complete=self._complete_request_state,
                fail=self._fail_request_state,
            ),
        )
        self.direct_tts_lock = threading.RLock()
        self.management_lock = threading.RLock()

    def _register_request_state(
        self,
        request_id: str,
        api_mode: str,
        backend: str,
        media_type: str,
        response_streaming: bool,
        deadline_ts: float | None = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> EngineRequestState:
        now = time.perf_counter()
        state = EngineRequestState(
            request_id=request_id,
            api_mode=api_mode,
            backend=backend,
            media_type=media_type,
            response_streaming=bool(response_streaming),
            submit_ts=now,
            deadline_ts=deadline_ts,
            updated_ts=now,
            meta=dict(meta or {}),
            lifecycle_timestamps={EngineStatus.NEW: now},
        )
        with self.request_registry_lock:
            self.active_requests[request_id] = state
        return state

    def _move_to_recent_locked(self, state: EngineRequestState) -> None:
        self.recent_requests.appendleft(state)
        while len(self.recent_requests) > self.recent_request_limit:
            self.recent_requests.pop()

    def _update_request_state(
        self,
        request_id: str,
        status: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.perf_counter()
        with self.request_registry_lock:
            state = self.active_requests.get(request_id)
            if state is None:
                return
            state.status = status
            state.updated_ts = now
            state.lifecycle_timestamps[status] = now
            if extra:
                backend = extra.pop("backend", None)
                if backend is not None:
                    state.backend = str(backend)
                finish_reason = extra.pop("finish_reason", None)
                if finish_reason is not None:
                    state.finish_reason = str(finish_reason)
                error = extra.pop("error", None)
                if error is not None:
                    state.error = str(error)
                state.profile.update(extra)

    def _merge_request_state_profile(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not extra:
            return
        now = time.perf_counter()
        with self.request_registry_lock:
            state = self.active_requests.get(request_id)
            if state is None:
                for recent_state in self.recent_requests:
                    if recent_state.request_id == request_id:
                        state = recent_state
                        break
            if state is None:
                return
            state.updated_ts = now
            backend = extra.get("backend")
            if backend is not None:
                state.backend = str(backend)
            finish_reason = extra.get("finish_reason")
            if finish_reason is not None:
                state.finish_reason = str(finish_reason)
            error = extra.get("error")
            if error is not None:
                state.error = str(error)
            merged = dict(extra)
            merged.pop("backend", None)
            merged.pop("finish_reason", None)
            merged.pop("error", None)
            state.profile.update(merged)

    def _complete_request_state(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        now = time.perf_counter()
        with self.request_registry_lock:
            state = self.active_requests.pop(request_id, None)
            if state is None:
                return
            state.status = EngineStatus.COMPLETED
            state.updated_ts = now
            state.lifecycle_timestamps[EngineStatus.COMPLETED] = now
            if extra:
                finish_reason = extra.pop("finish_reason", None)
                if finish_reason is not None:
                    state.finish_reason = str(finish_reason)
                state.profile.update(extra)
            self._move_to_recent_locked(state)

    def _fail_request_state(self, request_id: str, error: str) -> None:
        now = time.perf_counter()
        with self.request_registry_lock:
            state = self.active_requests.pop(request_id, None)
            if state is None:
                return
            state.status = EngineStatus.FAILED
            state.updated_ts = now
            state.error = str(error)
            state.lifecycle_timestamps[EngineStatus.FAILED] = now
            self._move_to_recent_locked(state)

    def _snapshot_request_registry(self) -> Dict[str, Any]:
        with self.request_registry_lock:
            active = [state.to_summary() for state in self.active_requests.values()]
            recent = [state.to_summary() for state in list(self.recent_requests)]
        active.sort(key=lambda item: item["submit_ts"])
        return {
            "active_count": len(active),
            "recent_count": len(recent),
            "recent_limit": self.recent_request_limit,
            "active_requests": active,
            "recent_requests": recent,
        }

    @staticmethod
    def _safe_component_snapshot(component: Any) -> Dict[str, Any] | None:
        if component is None or not hasattr(component, "snapshot"):
            return None
        try:
            return dict(component.snapshot())
        except Exception:
            return None

    def _build_stage_summary(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        active_requests = list(request_registry.get("active_requests", []))
        status_counts: Dict[str, int] = {}
        for item in active_requests:
            status = str(item.get("status", "UNKNOWN"))
            status_counts[status] = status_counts.get(status, 0) + 1

        bert_worker_state = self._safe_component_snapshot(getattr(self.tts, "prepare_bert_batch_worker", None))
        ref_semantic_worker_state = self._safe_component_snapshot(getattr(self.tts, "prepare_ref_semantic_batch_worker", None))
        text_preprocessor_state = self._safe_component_snapshot(getattr(self.tts, "text_preprocessor", None))

        return {
            "active_request_count": int(len(active_requests)),
            "status_counts": status_counts,
            "queued_request_count": int(status_counts.get(EngineStatus.QUEUED, 0)),
            "cpu_prepare_request_count": int(status_counts.get(EngineStatus.CPU_PREPARING, 0)),
            "gpu_prepare_request_count": int(status_counts.get(EngineStatus.GPU_PREPARING, 0)),
            "ready_for_prefill_request_count": int(status_counts.get(EngineStatus.READY_FOR_PREFILL, 0)),
            "active_decode_request_count": int(status_counts.get(EngineStatus.ACTIVE_DECODE, 0)),
            "ready_for_finalize_request_count": int(status_counts.get(EngineStatus.READY_FOR_FINALIZE, 0)),
            "finalizing_request_count": int(status_counts.get(EngineStatus.FINALIZING, 0)),
            "streaming_request_count": int(status_counts.get(EngineStatus.STREAMING, 0)),
            "worker_pending_jobs": int(worker_state.get("pending_jobs", 0)),
            "worker_decode_active_size": int(worker_state.get("running_requests", 0)),
            "worker_prepare_inflight": int(worker_state.get("prepare_inflight", 0)),
            "worker_finalize_pending": int(worker_state.get("finalize_pending", 0)),
            "worker_finalize_inflight": int(worker_state.get("finalize_inflight", 0)),
            "admission_config": {
                "decode_backlog_max": int(worker_state.get("decode_backlog_max", 0)),
                "finalize_pending_max": int(worker_state.get("finalize_pending_max", 0)),
            },
            "active_batch": dict(worker_state.get("active_batch") or {}),
            "prepare_state": dict(worker_state.get("prepare_state") or {}),
            "bert_batch_worker_state": bert_worker_state,
            "ref_semantic_worker_state": ref_semantic_worker_state,
            "text_preprocessor_state": text_preprocessor_state,
        }

    def _collect_request_summaries(self, request_ids: Sequence[str]) -> List[Dict[str, Any]]:
        requested = set(request_ids)
        results: List[Dict[str, Any]] = []
        with self.request_registry_lock:
            for state in self.active_requests.values():
                if state.request_id in requested:
                    results.append(state.to_summary())
            for state in self.recent_requests:
                if state.request_id in requested and all(item["request_id"] != state.request_id for item in results):
                    results.append(state.to_summary())
        results.sort(key=lambda item: item["request_id"])
        return results

    def _has_active_request(self, request_id: str) -> bool:
        with self.request_registry_lock:
            return request_id in self.active_requests

    @staticmethod
    def _build_request_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text")
        prompt_text = payload.get("prompt_text")
        return {
            "text_len": 0 if text is None else len(str(text)),
            "prompt_text_len": 0 if prompt_text is None else len(str(prompt_text)),
            "text_lang": payload.get("text_lang"),
            "prompt_lang": payload.get("prompt_lang"),
            "ref_audio_path": payload.get("ref_audio_path"),
        }

    @staticmethod
    def _sum_profile_field(items: Sequence[Dict[str, Any]], key: str) -> float:
        total = 0.0
        for item in items:
            value = item.get(key, 0.0)
            if isinstance(value, (int, float)):
                total += float(value)
        return total

    def _build_direct_segment_trace(
        self,
        segment_texts: Sequence[str],
        prepare_profiles: Sequence[Dict[str, Any]],
        worker_profiles: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for index, segment_text in enumerate(segment_texts):
            prepare_item = prepare_profiles[index] if index < len(prepare_profiles) else {}
            worker_item = worker_profiles[index] if index < len(worker_profiles) else {}
            prepare_profile = dict(prepare_item.get("prepare_profile", {}))
            results.append(
                {
                    "segment_index": index,
                    "request_id": prepare_item.get("request_id") or worker_item.get("request_id"),
                    "text_len": len(str(segment_text)),
                    "prepare_wall_ms": float(prepare_item.get("prepare_wall_ms", 0.0)),
                    "prepare_profile_total_ms": float(prepare_item.get("prepare_profile_total_ms", 0.0)),
                    "decode_admission_wait_ms": float(worker_item.get("decode_admission_wait_ms", 0.0)),
                    "queue_wait_ms": float(worker_item.get("queue_wait_ms", 0.0)),
                    "prefill_ms": float(worker_item.get("prefill_ms", 0.0)),
                    "merge_ms": float(worker_item.get("merge_ms", 0.0)),
                    "decode_ms": float(worker_item.get("decode_ms", 0.0)),
                    "finalize_wait_ms": float(worker_item.get("finalize_wait_ms", 0.0)),
                    "synth_ms": float(worker_item.get("synth_ms", 0.0)),
                    "worker_total_ms": float(worker_item.get("worker_total_ms", 0.0)),
                    "decode_steps": int(worker_item.get("decode_steps", 0)),
                    "semantic_len": int(worker_item.get("semantic_len", 0)),
                    "finish_reason": worker_item.get("finish_reason"),
                    "norm_text": prepare_profile.get("norm_text"),
                }
            )
        return results

    def _build_direct_scheduler_profile(
        self,
        *,
        backend: str,
        request_start: float,
        response_ready_at: float,
        audio_bytes: int,
        sample_rate: int,
        segment_texts: Sequence[str],
        prepare_profiles: Sequence[Dict[str, Any]],
        worker_profiles: Sequence[Dict[str, Any]],
        pack_ms: float,
        response_overhead_ms: float,
    ) -> Dict[str, Any]:
        segment_trace = self._build_direct_segment_trace(segment_texts, prepare_profiles, worker_profiles)
        prepare_profile_dicts = [dict(item.get("prepare_profile", {})) for item in prepare_profiles]
        request_total_ms = max(0.0, (response_ready_at - request_start) * 1000.0)
        prepare_wall_ms = self._sum_profile_field(prepare_profiles, "prepare_wall_ms")
        prepare_profile_total_ms = self._sum_profile_field(prepare_profiles, "prepare_profile_total_ms")
        decode_admission_wait_ms = self._sum_profile_field(worker_profiles, "decode_admission_wait_ms")
        queue_wait_ms = self._sum_profile_field(worker_profiles, "queue_wait_ms")
        prefill_ms = self._sum_profile_field(worker_profiles, "prefill_ms")
        merge_ms = self._sum_profile_field(worker_profiles, "merge_ms")
        decode_ms = self._sum_profile_field(worker_profiles, "decode_ms")
        finalize_wait_ms = self._sum_profile_field(worker_profiles, "finalize_wait_ms")
        synth_ms = self._sum_profile_field(worker_profiles, "synth_ms")
        worker_total_ms = self._sum_profile_field(worker_profiles, "worker_total_ms")
        decode_steps = sum(int(item.get("decode_steps", 0)) for item in worker_profiles)
        semantic_len = sum(int(item.get("semantic_len", 0)) for item in worker_profiles)
        request_other_ms = max(
            0.0,
            request_total_ms - prepare_wall_ms - worker_total_ms - pack_ms - response_overhead_ms,
        )
        return {
            "backend": backend,
            "backend_mode": backend,
            "segment_count": len(segment_texts),
            "sample_rate": int(sample_rate),
            "audio_bytes": int(audio_bytes),
            "request_total_ms": request_total_ms,
            "prepare_ms": prepare_wall_ms,
            "prepare_wall_ms": prepare_wall_ms,
            "prepare_profile_total_ms": prepare_profile_total_ms,
            "decode_admission_wait_ms": decode_admission_wait_ms,
            "queue_wait_ms": queue_wait_ms,
            "prefill_ms": prefill_ms,
            "merge_ms": merge_ms,
            "decode_ms": decode_ms,
            "finalize_wait_ms": finalize_wait_ms,
            "synth_ms": synth_ms,
            "pack_ms": pack_ms,
            "response_overhead_ms": response_overhead_ms,
            "worker_total_ms": worker_total_ms,
            "request_other_ms": request_other_ms,
            "decode_steps": decode_steps,
            "semantic_len": semantic_len,
            "prepare_segments": list(prepare_profiles),
            "worker_segments": list(worker_profiles),
            "segment_trace": segment_trace,
            "prepare_aggregate": self._aggregate_numeric_dicts(prepare_profile_dicts),
        }

    def _build_legacy_direct_profile(
        self,
        *,
        backend: str,
        fallback_reason: str | None,
        request_start: float,
        finished_at: float,
        sample_rate: int | None = None,
        audio_bytes: int = 0,
        pack_ms: float = 0.0,
        chunk_count: int = 0,
        stream_total_bytes: int = 0,
        first_chunk_ms: float | None = None,
    ) -> Dict[str, Any]:
        request_total_ms = max(0.0, (finished_at - request_start) * 1000.0)
        legacy_infer_ms = max(0.0, request_total_ms - pack_ms)
        return {
            "backend": backend,
            "backend_mode": backend,
            "fallback_reason": fallback_reason,
            "request_total_ms": request_total_ms,
            "prepare_ms": 0.0,
            "queue_wait_ms": 0.0,
            "prefill_ms": 0.0,
            "merge_ms": 0.0,
            "decode_ms": 0.0,
            "finalize_wait_ms": 0.0,
            "synth_ms": 0.0,
            "pack_ms": pack_ms,
            "worker_total_ms": legacy_infer_ms,
            "request_other_ms": 0.0,
            "legacy_infer_ms": legacy_infer_ms,
            "sample_rate": int(sample_rate) if sample_rate is not None else None,
            "audio_bytes": int(audio_bytes),
            "chunk_count": int(chunk_count),
            "stream_total_bytes": int(stream_total_bytes),
            "first_chunk_ms": None if first_chunk_ms is None else float(first_chunk_ms),
        }

    def _build_scheduler_submit_profile(
        self,
        *,
        backend: str,
        request_start: float,
        response_ready_at: float,
        audio_bytes: int,
        sample_rate: int,
        prepare_spec_build_ms: float,
        prepare_wall_ms: float,
        prepare_executor_queue_ms: float,
        prepare_executor_run_ms: float,
        prepare_profile_total_ms: float,
        prepare_profile_wall_ms: float,
        prepare_other_ms: float,
        api_after_prepare_ms: float,
        api_wait_result_ms: float,
        pack_ms: float,
        response_overhead_ms: float,
        worker_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        worker_total_ms = float(worker_profile.get("worker_total_ms", 0.0))
        request_total_ms = max(0.0, (response_ready_at - request_start) * 1000.0)
        request_other_ms = max(
            0.0,
            request_total_ms - prepare_wall_ms - api_after_prepare_ms - worker_total_ms - api_wait_result_ms - pack_ms,
        )
        result = {
            "backend": backend,
            "backend_mode": backend,
            "audio_bytes": int(audio_bytes),
            "sample_rate": int(sample_rate),
            "prepare_spec_build_ms": prepare_spec_build_ms,
            "prepare_ms": prepare_wall_ms,
            "prepare_wall_ms": prepare_wall_ms,
            "prepare_executor_queue_ms": prepare_executor_queue_ms,
            "prepare_executor_run_ms": prepare_executor_run_ms,
            "prepare_profile_total_ms": prepare_profile_total_ms,
            "prepare_profile_wall_ms": prepare_profile_wall_ms,
            "prepare_other_ms": prepare_other_ms,
            "api_after_prepare_ms": api_after_prepare_ms,
            "api_wait_result_ms": api_wait_result_ms,
            "pack_ms": pack_ms,
            "response_overhead_ms": response_overhead_ms,
            "request_total_ms": request_total_ms,
            "request_other_ms": request_other_ms,
        }
        result.update({key: value for key, value in worker_profile.items()})
        return result

    @staticmethod
    def _format_ms_header(value: Any) -> str:
        return f"{float(value):.3f}"

    def _build_scheduler_submit_headers(
        self,
        *,
        request_id: str,
        media_type: str,
        sample_rate: int,
        profile: Dict[str, Any],
    ) -> Dict[str, str]:
        prepare_profile = dict(profile.get("prepare_profile", {}))
        headers = {
            "X-Request-Id": request_id,
            "X-Semantic-Len": str(int(profile.get("semantic_len", 0))),
            "X-Finish-Reason": str(profile.get("finish_reason", "unknown")),
            "X-Queue-Wait-Ms": self._format_ms_header(profile.get("queue_wait_ms", 0.0)),
            "X-Decode-Admission-Wait-Ms": self._format_ms_header(profile.get("decode_admission_wait_ms", 0.0)),
            "X-Prepare-Ms": self._format_ms_header(profile.get("prepare_wall_ms", 0.0)),
            "X-Prepare-Wall-Ms": self._format_ms_header(profile.get("prepare_wall_ms", 0.0)),
            "X-Prepare-Spec-Build-Ms": self._format_ms_header(profile.get("prepare_spec_build_ms", 0.0)),
            "X-Prepare-Executor-Queue-Ms": self._format_ms_header(profile.get("prepare_executor_queue_ms", 0.0)),
            "X-Prepare-Admission-Wait-Ms": self._format_ms_header(prepare_profile.get("prepare_admission_wait_ms", 0.0)),
            "X-Prepare-Executor-Run-Ms": self._format_ms_header(profile.get("prepare_executor_run_ms", 0.0)),
            "X-Prepare-Profile-Total-Ms": self._format_ms_header(profile.get("prepare_profile_total_ms", 0.0)),
            "X-Prepare-Profile-Wall-Ms": self._format_ms_header(profile.get("prepare_profile_wall_ms", 0.0)),
            "X-Prepare-Other-Ms": self._format_ms_header(profile.get("prepare_other_ms", 0.0)),
            "X-Api-After-Prepare-Ms": self._format_ms_header(profile.get("api_after_prepare_ms", 0.0)),
            "X-Prefill-Ms": self._format_ms_header(profile.get("prefill_ms", 0.0)),
            "X-Merge-Ms": self._format_ms_header(profile.get("merge_ms", 0.0)),
            "X-Decode-Ms": self._format_ms_header(profile.get("decode_ms", 0.0)),
            "X-Finalize-Wait-Ms": self._format_ms_header(profile.get("finalize_wait_ms", 0.0)),
            "X-Synth-Ms": self._format_ms_header(profile.get("synth_ms", 0.0)),
            "X-Worker-Residual-Ms": self._format_ms_header(profile.get("worker_residual_ms", 0.0)),
            "X-Worker-Other-Ms": self._format_ms_header(profile.get("worker_other_ms", 0.0)),
            "X-Pack-Ms": self._format_ms_header(profile.get("pack_ms", 0.0)),
            "X-Worker-Total-Ms": self._format_ms_header(profile.get("worker_total_ms", 0.0)),
            "X-Api-Wait-Result-Ms": self._format_ms_header(profile.get("api_wait_result_ms", 0.0)),
            "X-Decode-Steps": str(int(profile.get("decode_steps", 0))),
            "X-Sample-Rate": str(int(sample_rate)),
            "X-Response-Overhead-Ms": self._format_ms_header(profile.get("response_overhead_ms", 0.0)),
            "X-Request-Other-Ms": self._format_ms_header(profile.get("request_other_ms", 0.0)),
            "X-Request-Total-Ms": self._format_ms_header(profile.get("request_total_ms", 0.0)),
        }
        headers.update(
            {
                "X-Prepare-Prompt-Text-Ms": self._format_ms_header(prepare_profile.get("prompt_text_features_ms", 0.0)),
                "X-Prepare-Target-Text-Ms": self._format_ms_header(prepare_profile.get("text_features_ms", 0.0)),
                "X-Prepare-Prompt-Text-CPU-Preprocess-Ms": self._format_ms_header(prepare_profile.get("prompt_text_cpu_preprocess_ms", 0.0)),
                "X-Prepare-Target-Text-CPU-Preprocess-Ms": self._format_ms_header(prepare_profile.get("text_cpu_preprocess_ms", 0.0)),
                "X-Prepare-Prompt-Text-CPU-Queue-Ms": self._format_ms_header(prepare_profile.get("prompt_text_cpu_queue_ms", 0.0)),
                "X-Prepare-Target-Text-CPU-Queue-Ms": self._format_ms_header(prepare_profile.get("text_cpu_queue_ms", 0.0)),
                "X-Prepare-Prompt-Text-Feature-Queue-Ms": self._format_ms_header(prepare_profile.get("prompt_text_feature_queue_ms", 0.0)),
                "X-Prepare-Target-Text-Feature-Queue-Ms": self._format_ms_header(prepare_profile.get("text_feature_queue_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_wait_ms", 0.0)),
                "X-Prepare-Target-Bert-Wait-Ms": self._format_ms_header(prepare_profile.get("text_bert_wait_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Admission-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_admission_wait_ms", 0.0)),
                "X-Prepare-Target-Bert-Admission-Wait-Ms": self._format_ms_header(prepare_profile.get("text_bert_admission_wait_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Queue-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_queue_wait_ms", 0.0)),
                "X-Prepare-Target-Bert-Queue-Wait-Ms": self._format_ms_header(prepare_profile.get("text_bert_queue_wait_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Batch-Collect-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_batch_collect_wait_ms", 0.0)),
                "X-Prepare-Target-Bert-Batch-Collect-Wait-Ms": self._format_ms_header(prepare_profile.get("text_bert_batch_collect_wait_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Forward-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_forward_ms", 0.0)),
                "X-Prepare-Target-Bert-Forward-Ms": self._format_ms_header(prepare_profile.get("text_bert_forward_ms", 0.0)),
                "X-Prepare-Prompt-Bert-Pending-On-Enqueue-Peak": str(int(prepare_profile.get("prompt_text_bert_pending_depth_on_enqueue_peak", 0.0))),
                "X-Prepare-Target-Bert-Pending-On-Enqueue-Peak": str(int(prepare_profile.get("text_bert_pending_depth_on_enqueue_peak", 0.0))),
                "X-Prepare-Prompt-Bert-Pending-On-Collect-Peak": str(int(prepare_profile.get("prompt_text_bert_pending_depth_on_collect_peak", 0.0))),
                "X-Prepare-Target-Bert-Pending-On-Collect-Peak": str(int(prepare_profile.get("text_bert_pending_depth_on_collect_peak", 0.0))),
                "X-Prepare-Prompt-Bert-High-Pressure-Peak": str(int(prepare_profile.get("prompt_text_bert_high_pressure_mode_peak", 0.0))),
                "X-Prepare-Target-Bert-High-Pressure-Peak": str(int(prepare_profile.get("text_bert_high_pressure_mode_peak", 0.0))),
                "X-Prepare-Prompt-Bert-Batch-Window-Ms": self._format_ms_header(prepare_profile.get("prompt_text_bert_batch_window_ms", 0.0)),
                "X-Prepare-Target-Bert-Batch-Window-Ms": self._format_ms_header(prepare_profile.get("text_bert_batch_window_ms", 0.0)),
                "X-Prepare-Text-Pair-Wall-Ms": self._format_ms_header(prepare_profile.get("text_feature_pair_ms", 0.0)),
                "X-Prepare-Text-CPU-Workers": str(int(prepare_profile.get("text_cpu_parallel_workers", 0.0))),
                "X-Prepare-Audio-Load-Ms": self._format_ms_header(prepare_profile.get("audio_load_ms", 0.0)),
                "X-Prepare-Audio-Stage-Wait-Ms": self._format_ms_header(prepare_profile.get("audio_stage_wait_ms", 0.0)),
                "X-Prepare-Prompt-Semantic-Ms": self._format_ms_header(prepare_profile.get("prompt_semantic_ms", 0.0)),
                "X-Prepare-Prompt-Semantic-Wait-Ms": self._format_ms_header(prepare_profile.get("prompt_semantic_wait_ms", 0.0)),
                "X-Prepare-Prompt-Semantic-CPU-Ms": self._format_ms_header(prepare_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)),
                "X-Prepare-Prompt-Semantic-Forward-Ms": self._format_ms_header(prepare_profile.get("prompt_semantic_forward_ms", 0.0)),
                "X-Prepare-Ref-Spec-Ms": self._format_ms_header(prepare_profile.get("ref_spec_ms", 0.0)),
                "X-Prepare-Ref-Spec-Wait-Ms": self._format_ms_header(prepare_profile.get("ref_spec_wait_ms", 0.0)),
                "X-Prepare-Ref-Bundle-Ms": self._format_ms_header(prepare_profile.get("ref_audio_bundle_ms", 0.0)),
                "X-Prepare-Tensorize-Ms": self._format_ms_header(prepare_profile.get("tensorize_ms", 0.0)),
                "X-Prepare-Inflight-On-Enter": str(int(prepare_profile.get("worker_prepare_inflight_on_enter", 0.0))),
                "X-Prepare-Inflight-Peak": str(int(prepare_profile.get("worker_prepare_peak_inflight", 0.0))),
            }
        )
        return headers

    def _build_scheduler_debug_request_profile(
        self,
        *,
        state: T2SRequestState,
        item: T2SFinishedItem,
        batch_request_count: int,
        prepare_batch_wall_ms: float,
        decode_batch_wall_ms: float,
        batch_request_total_ms: float,
    ) -> Dict[str, Any]:
        prepare_profile = dict(state.prepare_profile)
        prepare_wall_ms = float(prepare_profile.get("wall_total_ms", 0.0))
        return {
            "backend": "scheduler_debug",
            "backend_mode": "scheduler_debug",
            "batch_request_count": int(batch_request_count),
            "batch_prepare_wall_ms": float(prepare_batch_wall_ms),
            "batch_decode_wall_ms": float(decode_batch_wall_ms),
            "batch_request_total_ms": float(batch_request_total_ms),
            "prepare_ms": prepare_wall_ms,
            "prepare_wall_ms": prepare_wall_ms,
            "prepare_profile_total_ms": float(prepare_profile.get("wall_total_ms", prepare_wall_ms)),
            "prepare_profile": prepare_profile,
            "decode_steps": int(item.finish_idx),
            "finish_idx": int(item.finish_idx),
            "semantic_len": int(item.semantic_tokens.shape[0]),
            "finish_reason": item.finish_reason,
            "norm_text": state.norm_text,
            "norm_prompt_text": state.norm_prompt_text,
        }

    @staticmethod
    def _build_scheduler_debug_batch_profile(
        *,
        request_count: int,
        max_steps: int,
        prepare_batch_wall_ms: float,
        decode_batch_wall_ms: float,
        request_total_ms: float,
        finished_items: Sequence[T2SFinishedItem],
    ) -> Dict[str, Any]:
        finish_reason_counts: Dict[str, int] = {}
        total_semantic_len = 0
        for item in finished_items:
            finish_reason_counts[item.finish_reason] = finish_reason_counts.get(item.finish_reason, 0) + 1
            total_semantic_len += int(item.semantic_tokens.shape[0])
        return {
            "request_count": int(request_count),
            "max_steps": int(max_steps),
            "prepare_batch_wall_ms": float(prepare_batch_wall_ms),
            "decode_batch_wall_ms": float(decode_batch_wall_ms),
            "request_total_ms": float(request_total_ms),
            "total_semantic_len": int(total_semantic_len),
            "finish_reason_counts": finish_reason_counts,
        }

    def _normalize_lang(self, value: str | None) -> str | None:
        if value in [None, ""]:
            return value
        return str(value).lower()

    @staticmethod
    def _aggregate_numeric_dicts(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for item in items:
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value)
        return totals

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
    def _base_request_defaults() -> Dict[str, Any]:
        return {
            "request_id": None,
            "text": None,
            "text_lang": None,
            "ref_audio_path": None,
            "aux_ref_audio_paths": None,
            "prompt_text": "",
            "prompt_lang": None,
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "speed_factor": 1.0,
            "split_bucket": False,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "return_fragment": False,
            "fixed_length_chunk": False,
            "response_streaming": False,
            "parallel_infer": False,
            "repetition_penalty": 1.35,
            "sample_steps": 32,
            "super_sampling": False,
            "overlap_length": 2,
            "min_chunk_length": 16,
            "early_stop_num": -1,
            "ready_step": 0,
            "timeout_sec": None,
        }

    def _normalize_engine_request(
        self,
        payload: dict | NormalizedEngineRequest,
        *,
        request_id: str | None = None,
        normalize_streaming: bool = False,
        error_prefix: str = "request 参数非法: ",
    ) -> NormalizedEngineRequest:
        if isinstance(payload, NormalizedEngineRequest):
            normalized_payload = payload.to_payload()
        else:
            normalized_payload = self._base_request_defaults()
            normalized_payload.update(dict(payload))
        if request_id not in [None, ""]:
            normalized_payload["request_id"] = str(request_id)
        elif normalized_payload.get("request_id") in [None, ""]:
            raise ValueError("request_id is required after normalization")
        normalized_payload = self._apply_default_reference(normalized_payload)
        if normalize_streaming:
            normalized_payload = self._normalize_streaming_mode(normalized_payload)
        error = self.check_params(normalized_payload)
        if error is not None:
            raise ValueError(f"{error_prefix}{error}")
        timeout_sec = normalized_payload.get("timeout_sec")
        if timeout_sec in [None, ""]:
            parsed_timeout = None
        else:
            parsed_timeout = float(timeout_sec)
        aux_ref_audio_paths = normalized_payload.get("aux_ref_audio_paths")
        if aux_ref_audio_paths in [None, "", []]:
            normalized_aux_ref_audio_paths = None
        else:
            normalized_aux_ref_audio_paths = [str(item) for item in aux_ref_audio_paths]
        return NormalizedEngineRequest(
            request_id=str(normalized_payload["request_id"]),
            text=str(normalized_payload["text"]),
            text_lang=str(normalized_payload["text_lang"]),
            ref_audio_path=str(normalized_payload["ref_audio_path"]),
            prompt_lang=str(normalized_payload["prompt_lang"]),
            prompt_text="" if normalized_payload.get("prompt_text") is None else str(normalized_payload.get("prompt_text")),
            aux_ref_audio_paths=normalized_aux_ref_audio_paths,
            top_k=int(normalized_payload["top_k"]),
            top_p=float(normalized_payload["top_p"]),
            temperature=float(normalized_payload["temperature"]),
            repetition_penalty=float(normalized_payload["repetition_penalty"]),
            early_stop_num=int(normalized_payload.get("early_stop_num", -1)),
            ready_step=int(normalized_payload.get("ready_step", 0)),
            text_split_method=str(normalized_payload["text_split_method"]),
            batch_size=int(normalized_payload["batch_size"]),
            batch_threshold=float(normalized_payload["batch_threshold"]),
            split_bucket=bool(normalized_payload["split_bucket"]),
            speed_factor=float(normalized_payload["speed_factor"]),
            fragment_interval=float(normalized_payload["fragment_interval"]),
            seed=int(normalized_payload["seed"]),
            media_type=str(normalized_payload["media_type"]),
            streaming_mode=normalized_payload["streaming_mode"],
            return_fragment=bool(normalized_payload.get("return_fragment", False)),
            fixed_length_chunk=bool(normalized_payload.get("fixed_length_chunk", False)),
            response_streaming=bool(normalized_payload.get("response_streaming", False)),
            parallel_infer=bool(normalized_payload["parallel_infer"]),
            sample_steps=int(normalized_payload["sample_steps"]),
            super_sampling=bool(normalized_payload["super_sampling"]),
            overlap_length=int(normalized_payload["overlap_length"]),
            min_chunk_length=int(normalized_payload["min_chunk_length"]),
            timeout_sec=parsed_timeout,
        )

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

    @staticmethod
    def _is_aux_ref_enabled(aux_ref_audio_paths: List[str] | None) -> bool:
        return aux_ref_audio_paths not in [None, [], ()]

    def _select_direct_backend(self, normalized: NormalizedEngineRequest) -> Tuple[str, str | None]:
        if normalized.response_streaming:
            if normalized.return_fragment or normalized.fixed_length_chunk:
                return "legacy_direct_fragment", "fragment_streaming_mode"
            return "legacy_direct_streaming", "streaming_mode"
        if self._is_aux_ref_enabled(normalized.aux_ref_audio_paths):
            return "legacy_direct_aux_ref", "aux_ref_audio_paths"
        if normalized.super_sampling:
            return "legacy_direct_super_sampling", "super_sampling"
        if normalized.prompt_text in [None, ""]:
            return "legacy_direct_missing_prompt", "missing_prompt_text"
        return "scheduler_v1_direct", None

    def _iter_legacy_direct_tts_bytes(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> Generator[bytes, None, None]:
        payload = normalized.to_payload()
        media_type = normalized.media_type
        request_id = normalized.request_id
        request_start = time.perf_counter()
        chunk_count = 0
        stream_total_bytes = 0
        first_chunk_ms: float | None = None
        self._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        try:
            with self.direct_tts_lock:
                tts_generator = self.tts.run(payload)
                first_chunk = True
                current_media_type = media_type
                for sr, chunk in tts_generator:
                    if first_chunk:
                        first_chunk_ms = max(0.0, (time.perf_counter() - request_start) * 1000.0)
                        self._update_request_state(
                            request_id,
                            EngineStatus.STREAMING,
                            {
                                "backend": backend,
                                "backend_mode": backend,
                                "fallback_reason": fallback_reason,
                                "sample_rate": int(sr),
                            },
                        )
                    if first_chunk and media_type == "wav":
                        header = wave_header_chunk(sample_rate=sr)
                        chunk_count += 1
                        stream_total_bytes += len(header)
                        yield header
                        current_media_type = "raw"
                        first_chunk = False
                    elif first_chunk:
                        first_chunk = False
                    packed_chunk = pack_audio(BytesIO(), chunk, sr, current_media_type).getvalue()
                    chunk_count += 1
                    stream_total_bytes += len(packed_chunk)
                    yield packed_chunk
        except Exception as exc:
            self._fail_request_state(request_id, str(exc))
            raise
        self._complete_request_state(
            request_id,
            dict(
                self._build_legacy_direct_profile(
                    backend=backend,
                    fallback_reason=fallback_reason,
                    request_start=request_start,
                    finished_at=time.perf_counter(),
                    audio_bytes=stream_total_bytes,
                    chunk_count=chunk_count,
                    stream_total_bytes=stream_total_bytes,
                    first_chunk_ms=first_chunk_ms,
                ),
                streaming_completed=True,
            ),
        )

    def _should_use_scheduler_backend_for_direct(self, req: dict | NormalizedEngineRequest) -> bool:
        if isinstance(req, NormalizedEngineRequest):
            normalized = req
        else:
            normalized = self._normalize_engine_request(
                req,
                request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
                normalize_streaming=True,
            )
        backend, _ = self._select_direct_backend(normalized)
        return backend == "scheduler_v1_direct"

    def _segment_direct_text(self, normalized: dict | NormalizedEngineRequest) -> List[str]:
        payload = normalized.to_payload() if isinstance(normalized, NormalizedEngineRequest) else normalized
        return self.tts.text_preprocessor.pre_seg_text(
            str(payload["text"]),
            str(payload["text_lang"]),
            str(payload.get("text_split_method", "cut5")),
        )

    def _build_segment_request(
        self,
        normalized: NormalizedEngineRequest,
        *,
        request_id: str,
        text: str,
    ) -> NormalizedEngineRequest:
        payload = normalized.to_payload()
        payload["request_id"] = request_id
        payload["text"] = text
        payload["streaming_mode"] = False
        payload["return_fragment"] = False
        payload["fixed_length_chunk"] = False
        payload["response_streaming"] = False
        return self._normalize_engine_request(payload, error_prefix="segment request 参数非法: ")

    async def _run_direct_tts_via_scheduler(self, normalized: NormalizedEngineRequest) -> DirectTTSExecution:
        request_start = time.perf_counter()
        request_id = normalized.request_id
        media_type = normalized.media_type
        segment_texts = self._segment_direct_text(normalized)
        if not segment_texts:
            raise ValueError("text preprocessing returned no valid segments")
        self._update_request_state(
            request_id,
            EngineStatus.CPU_PREPARING,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct", "segment_count": len(segment_texts)},
        )
        segment_specs: List[SchedulerRequestSpec] = []
        for segment_index, segment_text in enumerate(segment_texts):
            segment_request = self._build_segment_request(
                normalized,
                request_id=f"{request_id}_seg_{segment_index:03d}",
                text=segment_text,
            )
            segment_specs.append(self.build_scheduler_submit_spec(segment_request))

        prepared_items = await asyncio.gather(
            *[
                self.scheduler_worker.prepare_state_profiled_async(spec, time.perf_counter())
                for spec in segment_specs
            ]
        )
        prepare_profiles: List[Dict[str, Any]] = []
        jobs: List[SchedulerPendingJob] = []
        loop = asyncio.get_running_loop()
        done_futures: List[asyncio.Future] = []
        for spec, (state, prepare_exec_started_at, prepare_exec_finished_at) in zip(segment_specs, prepared_items):
            prepare_wall_ms = max(0.0, (prepare_exec_finished_at - prepare_exec_started_at) * 1000.0)
            prepare_profile_total_ms = float(state.prepare_profile.get("wall_total_ms", prepare_wall_ms))
            prepare_profiles.append(
                {
                    "request_id": spec.request_id,
                    "prepare_wall_ms": prepare_wall_ms,
                    "prepare_profile_total_ms": prepare_profile_total_ms,
                    "prepare_profile": dict(state.prepare_profile),
                }
            )
            done_future = loop.create_future()
            done_futures.append(done_future)
            jobs.append(
                await self.scheduler_worker.submit_async(
                    state=state,
                    speed_factor=float(normalized.speed_factor),
                    sample_steps=int(normalized.sample_steps),
                    media_type=media_type,
                    prepare_wall_ms=prepare_wall_ms,
                    prepare_profile_total_ms=prepare_profile_total_ms,
                    done_loop=loop,
                    done_future=done_future,
                    engine_request_id=None,
                    timeout_sec=normalized.timeout_sec,
                )
            )
        self._update_request_state(
            request_id,
            EngineStatus.READY_FOR_PREFILL,
            {
                "backend": "scheduler_v1_direct",
                "backend_mode": "scheduler_v1_direct",
                "segment_count": len(segment_specs),
                "prepare_aggregate": self._aggregate_numeric_dicts(
                    [item["prepare_profile"] for item in prepare_profiles]
                ),
            },
        )
        self._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct"},
        )
        timeout_sec = float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0)
        await asyncio.wait_for(asyncio.gather(*done_futures), timeout=timeout_sec)

        sample_rate: int | None = None
        audio_parts: List[np.ndarray] = []
        worker_profiles: List[Dict[str, Any]] = []
        fragment_interval = float(normalized.fragment_interval)
        silence_chunk: Optional[np.ndarray] = None
        for job in jobs:
            if job.error is not None:
                raise RuntimeError(job.error)
            if job.audio_data is None or job.sample_rate is None or job.result is None:
                raise RuntimeError(f"{job.request_id} finished without audio result")
            if sample_rate is None:
                sample_rate = int(job.sample_rate)
                silence_samples = int(fragment_interval * float(sample_rate))
                if silence_samples > 0:
                    silence_chunk = np.zeros(silence_samples, dtype=np.int16)
            elif int(job.sample_rate) != sample_rate:
                raise RuntimeError("segment sample rate mismatch")
            audio_parts.append(job.audio_data)
            if silence_chunk is not None:
                audio_parts.append(silence_chunk.copy())
            worker_profiles.append(dict(job.result))
        if sample_rate is None or not audio_parts:
            raise RuntimeError("direct scheduler backend produced no audio")
        self._update_request_state(
            request_id,
            EngineStatus.FINALIZING,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct"},
        )
        merged_audio = np.concatenate(audio_parts, axis=0)
        pack_start = time.perf_counter()
        audio_bytes = pack_audio(BytesIO(), merged_audio, sample_rate, media_type).getvalue()
        pack_ms = max(0.0, (time.perf_counter() - pack_start) * 1000.0)
        direct_profile = self._build_direct_scheduler_profile(
            backend="scheduler_v1_direct",
            request_start=request_start,
            response_ready_at=time.perf_counter(),
            audio_bytes=len(audio_bytes),
            sample_rate=int(sample_rate),
            segment_texts=segment_texts,
            prepare_profiles=prepare_profiles,
            worker_profiles=worker_profiles,
            pack_ms=pack_ms,
            response_overhead_ms=0.0,
        )
        self._complete_request_state(
            request_id,
            dict(direct_profile, streaming_completed=False),
        )
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=audio_bytes,
            request_id=request_id,
        )

    def _run_legacy_direct_tts_blocking(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        normalized_payload = normalized.to_payload()
        request_id = normalized.request_id
        media_type = normalized.media_type
        request_start = time.perf_counter()
        self._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        with self.direct_tts_lock:
            tts_generator = self.tts.run(normalized_payload)
            try:
                sr, audio_data = next(tts_generator)
            except Exception as exc:
                self._fail_request_state(request_id, str(exc))
                raise
        self._update_request_state(
            request_id,
            EngineStatus.FINALIZING,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        pack_start = time.perf_counter()
        packed_audio = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
        pack_ms = max(0.0, (time.perf_counter() - pack_start) * 1000.0)
        self._complete_request_state(
            request_id,
            dict(
                self._build_legacy_direct_profile(
                    backend=backend,
                    fallback_reason=fallback_reason,
                    request_start=request_start,
                    finished_at=time.perf_counter(),
                    sample_rate=int(sr),
                    audio_bytes=len(packed_audio),
                    pack_ms=pack_ms,
                ),
                streaming_completed=False,
            ),
        )
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=packed_audio,
            request_id=request_id,
        )

    async def _run_direct_tts_via_legacy_backend(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        if normalized.response_streaming:
            return DirectTTSExecution(
                media_type=normalized.media_type,
                streaming=True,
                audio_generator=self._iter_legacy_direct_tts_bytes(
                    normalized,
                    backend=backend,
                    fallback_reason=fallback_reason,
                ),
                request_id=normalized.request_id,
            )
        return await asyncio.to_thread(
            self._run_legacy_direct_tts_blocking,
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    async def run_direct_tts_async(self, req: dict) -> DirectTTSExecution:
        normalized = self._normalize_engine_request(
            req,
            request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
            normalize_streaming=True,
            error_prefix="",
        )
        request_id = normalized.request_id
        media_type = normalized.media_type
        backend, fallback_reason = self._select_direct_backend(normalized)
        self._register_request_state(
            request_id=request_id,
            api_mode="tts",
            backend=backend,
            media_type=media_type,
            response_streaming=bool(normalized.response_streaming),
            deadline_ts=(
                time.perf_counter() + float(normalized.timeout_sec)
                if normalized.timeout_sec is not None
                else None
            ),
            meta=self._build_request_meta(normalized.to_payload()),
        )
        self._update_request_state(
            request_id,
            EngineStatus.VALIDATED,
            {
                "request_source": "direct_tts",
                "selected_backend": backend,
                "fallback_reason": fallback_reason,
            },
        )
        if backend == "scheduler_v1_direct":
            try:
                return await self._run_direct_tts_via_scheduler(normalized)
            except Exception as exc:
                self._fail_request_state(request_id, str(exc))
                raise
        return await self._run_direct_tts_via_legacy_backend(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    def run_direct_tts(self, req: dict) -> DirectTTSExecution:
        normalized = self._normalize_engine_request(
            req,
            request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
            normalize_streaming=True,
            error_prefix="",
        )
        request_id = normalized.request_id
        media_type = normalized.media_type
        backend, fallback_reason = self._select_direct_backend(normalized)
        if not self._has_active_request(request_id):
            self._register_request_state(
                request_id=request_id,
                api_mode="tts",
                backend=backend,
                media_type=media_type,
                response_streaming=bool(normalized.response_streaming),
                meta=self._build_request_meta(normalized.to_payload()),
            )
        self._update_request_state(
            request_id,
            EngineStatus.VALIDATED,
            {
                "request_source": "direct_tts",
                "selected_backend": backend,
                "fallback_reason": fallback_reason,
            },
        )
        if backend != "scheduler_v1_direct":
            if normalized.response_streaming:
                return DirectTTSExecution(
                    media_type=media_type,
                    streaming=True,
                    audio_generator=self._iter_legacy_direct_tts_bytes(
                        normalized,
                        backend=backend,
                        fallback_reason=fallback_reason,
                    ),
                    request_id=request_id,
                )
            return self._run_legacy_direct_tts_blocking(
                normalized,
                backend=backend,
                fallback_reason=fallback_reason,
            )
        normalized_payload = normalized.to_payload()
        if normalized.response_streaming:
            return DirectTTSExecution(
                media_type=media_type,
                streaming=True,
                audio_generator=self._iter_legacy_direct_tts_bytes(
                    normalized,
                    backend="legacy_direct_sync_compat",
                    fallback_reason="sync_direct_compat",
                ),
                request_id=request_id,
            )
        return self._run_legacy_direct_tts_blocking(
            normalized,
            backend="legacy_direct_sync_compat",
            fallback_reason="sync_direct_compat",
        )

    def build_scheduler_request_specs(self, request_items: List[dict]) -> List[SchedulerRequestSpec]:
        specs: List[SchedulerRequestSpec] = []
        for index, payload in enumerate(request_items):
            normalized = self._normalize_engine_request(
                payload,
                request_id=str(payload.get("request_id") or f"req_{index:03d}"),
                error_prefix=f"request[{index}] 参数非法: ",
            )
            specs.append(normalized.to_scheduler_spec())
        return specs

    def build_scheduler_submit_spec(self, payload: dict | NormalizedEngineRequest) -> SchedulerRequestSpec:
        normalized = self._normalize_engine_request(
            payload,
            request_id=(
                payload.request_id
                if isinstance(payload, NormalizedEngineRequest)
                else str(payload.get("request_id") or f"job_{uuid.uuid4().hex[:12]}")
            ),
        )
        return normalized.to_scheduler_spec()

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
        request_start = time.perf_counter()
        set_scheduler_seed(seed)
        specs = self.build_scheduler_request_specs(request_items)
        request_ids = [spec.request_id for spec in specs]
        for spec in specs:
            self._register_request_state(
                request_id=spec.request_id,
                api_mode="scheduler_debug",
                backend="scheduler_debug",
                media_type="wav",
                response_streaming=False,
                meta={
                    "text_len": len(spec.text),
                    "prompt_text_len": len(spec.prompt_text),
                    "text_lang": spec.text_lang,
                    "prompt_lang": spec.prompt_lang,
                    "ref_audio_path": str(spec.ref_audio_path),
                    "ready_step": int(spec.ready_step),
                },
            )
            self._update_request_state(spec.request_id, EngineStatus.VALIDATED, {"request_source": "scheduler_debug"})
            self._update_request_state(spec.request_id, EngineStatus.CPU_PREPARING, None)
        prepare_started_at = time.perf_counter()
        try:
            states = await self.scheduler_worker.prepare_states_batch_async(specs)
        except Exception as exc:
            for request_id in request_ids:
                self._fail_request_state(request_id, str(exc))
            raise
        prepare_finished_at = time.perf_counter()
        prepare_batch_wall_ms = max(0.0, (prepare_finished_at - prepare_started_at) * 1000.0)
        for state in states:
            self._update_request_state(
                state.request_id,
                EngineStatus.ACTIVE_DECODE,
                {
                    "prepare_profile": dict(state.prepare_profile),
                    "norm_text": state.norm_text,
                    "norm_prompt_text": state.norm_prompt_text,
                },
            )
        decode_started_at = time.perf_counter()
        try:
            finished = run_scheduler_continuous(self.tts.t2s_model.model, states, max_steps=int(max_steps))
        except Exception as exc:
            for request_id in request_ids:
                self._fail_request_state(request_id, str(exc))
            raise
        decode_finished_at = time.perf_counter()
        decode_batch_wall_ms = max(0.0, (decode_finished_at - decode_started_at) * 1000.0)
        request_total_ms = max(0.0, (decode_finished_at - request_start) * 1000.0)
        finished_map = {item.request_id: item for item in finished}
        request_profiles: List[Dict[str, Any]] = []
        for state in states:
            item = finished_map.get(state.request_id)
            if item is None:
                self._fail_request_state(state.request_id, "scheduler_debug finished without result")
                continue
            request_profile = self._build_scheduler_debug_request_profile(
                state=state,
                item=item,
                batch_request_count=len(states),
                prepare_batch_wall_ms=prepare_batch_wall_ms,
                decode_batch_wall_ms=decode_batch_wall_ms,
                batch_request_total_ms=request_total_ms,
            )
            request_profiles.append(
                {
                    "request_id": state.request_id,
                    "profile": dict(request_profile),
                }
            )
            self._complete_request_state(
                state.request_id,
                dict(request_profile),
            )
        return SchedulerDebugExecution(
            payload={
                "message": "success",
                "request_count": len(states),
                "max_steps": int(max_steps),
                "batch_profile": self._build_scheduler_debug_batch_profile(
                    request_count=len(states),
                    max_steps=int(max_steps),
                    prepare_batch_wall_ms=prepare_batch_wall_ms,
                    decode_batch_wall_ms=decode_batch_wall_ms,
                    request_total_ms=request_total_ms,
                    finished_items=finished,
                ),
                "requests": self.summarize_scheduler_states(states),
                "finished": self.summarize_scheduler_finished(finished),
                "request_profiles": request_profiles,
                "request_traces": self._collect_request_summaries(request_ids),
            }
        )

    async def run_scheduler_submit(self, payload: dict) -> SchedulerSubmitExecution:
        request_start = time.perf_counter()
        prepare_start = request_start
        normalized = self._normalize_engine_request(
            payload,
            request_id=str(payload.get("request_id") or f"job_{uuid.uuid4().hex[:12]}"),
        )
        spec = self.build_scheduler_submit_spec(normalized)
        deadline_ts = None
        timeout_sec = normalized.timeout_sec
        if timeout_sec is not None:
            try:
                deadline_ts = request_start + float(timeout_sec)
            except Exception:
                deadline_ts = None
        self._register_request_state(
            request_id=spec.request_id,
            api_mode="scheduler_submit",
            backend="scheduler_v1",
            media_type=normalized.media_type,
            response_streaming=False,
            deadline_ts=deadline_ts,
            meta=self._build_request_meta(normalized.to_payload()),
        )
        self._update_request_state(spec.request_id, EngineStatus.VALIDATED, {"request_source": "scheduler_submit"})
        spec_ready_at = time.perf_counter()
        prepare_spec_build_ms = max(0.0, (spec_ready_at - prepare_start) * 1000.0)
        self._update_request_state(spec.request_id, EngineStatus.CPU_PREPARING, {"prepare_spec_build_ms": prepare_spec_build_ms})
        try:
            state, prepare_exec_started_at, prepare_exec_finished_at = await self.scheduler_worker.prepare_state_profiled_async(
                spec,
                spec_ready_at,
            )
        except Exception as exc:
            self._fail_request_state(spec.request_id, str(exc))
            raise
        prepare_wall_ms = max(0.0, (prepare_exec_finished_at - spec_ready_at) * 1000.0)
        prepare_executor_queue_ms = max(0.0, (prepare_exec_started_at - spec_ready_at) * 1000.0)
        prepare_executor_run_ms = max(0.0, (prepare_exec_finished_at - prepare_exec_started_at) * 1000.0)
        prepare_profile = dict(state.prepare_profile)
        prepare_profile_total_ms = float(prepare_profile.get("wall_total_ms", prepare_wall_ms))
        prepare_profile_wall_ms = float(prepare_profile.get("wall_total_ms", prepare_wall_ms))
        prepare_other_ms = max(0.0, prepare_wall_ms - prepare_spec_build_ms - prepare_executor_queue_ms - prepare_executor_run_ms)
        self._update_request_state(
            spec.request_id,
            EngineStatus.READY_FOR_PREFILL,
            {
                "prepare_wall_ms": prepare_wall_ms,
                "prepare_profile_total_ms": prepare_profile_total_ms,
                "prepare_profile": prepare_profile,
            },
        )
        api_after_prepare_start = time.perf_counter()
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        job = await self.scheduler_worker.submit_async(
            state=state,
            speed_factor=float(normalized.speed_factor),
            sample_steps=int(normalized.sample_steps),
            media_type=normalized.media_type,
            prepare_wall_ms=prepare_wall_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            done_loop=loop,
            done_future=done_future,
            engine_request_id=spec.request_id,
            timeout_sec=normalized.timeout_sec,
        )
        api_after_prepare_ms = max(0.0, (time.perf_counter() - api_after_prepare_start) * 1000.0)
        try:
            await asyncio.wait_for(done_future, timeout=float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0))
        except Exception as exc:
            self._fail_request_state(spec.request_id, str(exc))
            raise
        wait_return_at = time.perf_counter()
        if job.error is not None:
            raise RuntimeError(job.error)
        if job.audio_data is None or job.sample_rate is None or job.result is None:
            self._fail_request_state(spec.request_id, f"{job.request_id} finished without audio result")
            raise RuntimeError(f"{job.request_id} finished without audio result")
        pack_start = time.perf_counter()
        audio_data = pack_audio(BytesIO(), job.audio_data, int(job.sample_rate), job.media_type).getvalue()
        pack_end = time.perf_counter()
        pack_ms = (pack_end - pack_start) * 1000.0
        api_wait_result_ms = 0.0
        if job.result_ready_time is not None:
            api_wait_result_ms = max(0.0, (wait_return_at - job.result_ready_time) * 1000.0)
        response_ready_at = time.perf_counter()
        response_overhead_ms = max(0.0, (response_ready_at - pack_end) * 1000.0)
        submit_profile = self._build_scheduler_submit_profile(
            backend="scheduler_v1",
            request_start=request_start,
            response_ready_at=response_ready_at,
            audio_bytes=len(audio_data),
            sample_rate=int(job.sample_rate),
            prepare_spec_build_ms=prepare_spec_build_ms,
            prepare_wall_ms=prepare_wall_ms,
            prepare_executor_queue_ms=prepare_executor_queue_ms,
            prepare_executor_run_ms=prepare_executor_run_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            prepare_profile_wall_ms=prepare_profile_wall_ms,
            prepare_other_ms=prepare_other_ms,
            api_after_prepare_ms=api_after_prepare_ms,
            api_wait_result_ms=api_wait_result_ms,
            pack_ms=pack_ms,
            response_overhead_ms=response_overhead_ms,
            worker_profile=dict(job.result or {}),
        )
        headers = self._build_scheduler_submit_headers(
            request_id=job.request_id,
            media_type=job.media_type,
            sample_rate=int(job.sample_rate),
            profile=submit_profile,
        )
        self._merge_request_state_profile(
            spec.request_id,
            dict(submit_profile, response_headers_emitted=True),
        )
        return SchedulerSubmitExecution(audio_bytes=audio_data, media_type=f"audio/{job.media_type}", headers=headers)

    def get_scheduler_state(self) -> dict:
        return self.scheduler_worker.snapshot()

    def get_runtime_state(self) -> dict:
        model_state = self.model_registry.snapshot()
        default_ref = self.reference_registry.get_default()
        scheduler_state = self.get_scheduler_state()
        request_registry = self._snapshot_request_registry()
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
            "request_registry": request_registry,
            "stage_summary": self._build_stage_summary(request_registry, scheduler_state),
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
