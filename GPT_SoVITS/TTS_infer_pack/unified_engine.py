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
from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import PrepareCoordinator, PreparedCpuStage
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


@dataclass
class EnginePolicyConfig:
    enabled: bool = True
    poll_wait_ms: float = 5.0
    decode_backlog_soft_max: int = 0
    finalize_pending_soft_max: int = 0
    prepare_inflight_soft_max: int = 0
    active_decode_soft_max: int = 0
    ready_for_prefill_soft_max: int = 0
    active_request_soft_max: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "poll_wait_ms": float(self.poll_wait_ms),
            "decode_backlog_soft_max": int(self.decode_backlog_soft_max),
            "finalize_pending_soft_max": int(self.finalize_pending_soft_max),
            "prepare_inflight_soft_max": int(self.prepare_inflight_soft_max),
            "active_decode_soft_max": int(self.active_decode_soft_max),
            "ready_for_prefill_soft_max": int(self.ready_for_prefill_soft_max),
            "active_request_soft_max": int(self.active_request_soft_max),
        }


@dataclass
class EngineArbiterConfig:
    poll_wait_ms: float = 5.0
    decode_burst: int = 4
    prepare_aging_ms: float = 10.0
    finalize_aging_ms: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "poll_wait_ms": float(self.poll_wait_ms),
            "decode_burst": int(self.decode_burst),
            "prepare_aging_ms": float(self.prepare_aging_ms),
            "finalize_aging_ms": float(self.finalize_aging_ms),
        }


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


class EngineRequestRegistry:
    def __init__(self, recent_limit: int) -> None:
        self.lock = threading.Lock()
        self.active_requests: Dict[str, EngineRequestState] = {}
        self.recent_requests: Deque[EngineRequestState] = deque()
        self.recent_limit = max(1, int(recent_limit))

    def register(
        self,
        *,
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
        with self.lock:
            self.active_requests[request_id] = state
        return state

    def _move_to_recent_locked(self, state: EngineRequestState) -> None:
        self.recent_requests.appendleft(state)
        while len(self.recent_requests) > self.recent_limit:
            self.recent_requests.pop()

    @staticmethod
    def _apply_state_extra(state: EngineRequestState, extra: Optional[Dict[str, Any]]) -> None:
        if not extra:
            return
        payload = dict(extra)
        backend = payload.pop("backend", None)
        if backend is not None:
            state.backend = str(backend)
        finish_reason = payload.pop("finish_reason", None)
        if finish_reason is not None:
            state.finish_reason = str(finish_reason)
        error = payload.pop("error", None)
        if error is not None:
            state.error = str(error)
        state.profile.update(payload)

    def update(self, request_id: str, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        now = time.perf_counter()
        with self.lock:
            state = self.active_requests.get(request_id)
            if state is None:
                return
            state.status = str(status)
            state.updated_ts = now
            state.lifecycle_timestamps[str(status)] = now
            self._apply_state_extra(state, extra)

    def merge_profile(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not extra:
            return
        now = time.perf_counter()
        with self.lock:
            state = self.active_requests.get(request_id)
            if state is None:
                for recent_state in self.recent_requests:
                    if recent_state.request_id == request_id:
                        state = recent_state
                        break
            if state is None:
                return
            state.updated_ts = now
            self._apply_state_extra(state, extra)

    def complete(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        now = time.perf_counter()
        with self.lock:
            state = self.active_requests.pop(request_id, None)
            if state is None:
                return
            state.status = EngineStatus.COMPLETED
            state.updated_ts = now
            state.lifecycle_timestamps[EngineStatus.COMPLETED] = now
            self._apply_state_extra(state, extra)
            self._move_to_recent_locked(state)

    def fail(self, request_id: str, error: str) -> None:
        now = time.perf_counter()
        with self.lock:
            state = self.active_requests.pop(request_id, None)
            if state is None:
                return
            state.status = EngineStatus.FAILED
            state.updated_ts = now
            state.error = str(error)
            state.lifecycle_timestamps[EngineStatus.FAILED] = now
            self._move_to_recent_locked(state)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            active = [state.to_summary() for state in self.active_requests.values()]
            recent = [state.to_summary() for state in list(self.recent_requests)]
            recent_limit = self.recent_limit
        active.sort(key=lambda item: item["submit_ts"])
        return {
            "active_count": len(active),
            "recent_count": len(recent),
            "recent_limit": recent_limit,
            "active_requests": active,
            "recent_requests": recent,
        }

    def collect_summaries(self, request_ids: Sequence[str]) -> List[Dict[str, Any]]:
        requested = set(request_ids)
        results: List[Dict[str, Any]] = []
        with self.lock:
            for state in self.active_requests.values():
                if state.request_id in requested:
                    results.append(state.to_summary())
            existing_ids = {item["request_id"] for item in results}
            for state in self.recent_requests:
                if state.request_id in requested and state.request_id not in existing_ids:
                    results.append(state.to_summary())
        results.sort(key=lambda item: item["request_id"])
        return results

    def has_active(self, request_id: str) -> bool:
        with self.lock:
            return request_id in self.active_requests


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
    engine_policy_wait_ms: float = 0.0
    engine_dispatch_wait_ms: float = 0.0
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


class SchedulerJobRegistry:
    def __init__(self, lock: threading.Lock | threading.RLock | threading.Condition) -> None:
        self._lock = lock
        self._job_map: Dict[str, SchedulerPendingJob] = {}
        self._total_submitted = 0
        self._total_finished = 0

    def register(self, job: SchedulerPendingJob, *, keep_job: bool = True) -> None:
        with self._lock:
            if keep_job:
                self._job_map[job.request_id] = job
            self._total_submitted += 1

    def get(self, request_id: str) -> SchedulerPendingJob | None:
        with self._lock:
            return self._job_map.get(request_id)

    def pop(self, request_id: str) -> SchedulerPendingJob | None:
        with self._lock:
            return self._job_map.pop(request_id, None)

    def remove(self, request_id: str) -> None:
        with self._lock:
            self._job_map.pop(request_id, None)

    def mark_finished(self) -> None:
        with self._lock:
            self._total_finished += 1

    def mark_finished_and_remove(self, request_id: str) -> None:
        with self._lock:
            self._job_map.pop(request_id, None)
            self._total_finished += 1

    def is_empty(self) -> bool:
        with self._lock:
            return not self._job_map

    def submitted_count(self) -> int:
        with self._lock:
            return int(self._total_submitted)

    def finished_count(self) -> int:
        with self._lock:
            return int(self._total_finished)

    def snapshot(self, max_request_ids: int = 32) -> Dict[str, Any]:
        with self._lock:
            request_ids = list(self._job_map.keys())
            return {
                "job_count": int(len(request_ids)),
                "request_ids": request_ids[: max(0, int(max_request_ids))],
                "total_submitted": int(self._total_submitted),
                "total_finished": int(self._total_finished),
            }


class EngineTaskQueueOwner:
    def __init__(self, completion_key: str = "total_completed") -> None:
        self.condition = threading.Condition()
        self.queue: Deque[Any] = deque()
        self.total_submitted = 0
        self.total_completed = 0
        self.peak_waiting = 0
        self.completion_key = str(completion_key)

    def enqueue(self, item: Any) -> None:
        with self.condition:
            self.queue.append(item)
            self.total_submitted += 1
            self.peak_waiting = max(self.peak_waiting, len(self.queue))
            self.condition.notify_all()

    def enqueue_many(self, items: Sequence[Any]) -> None:
        if not items:
            return
        with self.condition:
            for item in items:
                self.queue.append(item)
            self.total_submitted += len(items)
            self.peak_waiting = max(self.peak_waiting, len(self.queue))
            self.condition.notify_all()

    def pop_left(self) -> Any | None:
        with self.condition:
            if not self.queue:
                return None
            return self.queue.popleft()

    def mark_completed(self, count: int = 1, *, notify: bool = False) -> None:
        if count <= 0:
            return
        with self.condition:
            self.total_completed += int(count)
            if notify:
                self.condition.notify_all()

    def has_items(self) -> bool:
        with self.condition:
            return bool(self.queue)

    def waiting_count(self) -> int:
        with self.condition:
            return int(len(self.queue))

    def snapshot(self, *, max_request_ids: int = 16, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.condition:
            waiting_items = list(self.queue)[: max(0, int(max_request_ids))]
            snapshot = {
                "waiting_count": int(len(self.queue)),
                "waiting_request_ids": [str(getattr(item, "request_id", "")) for item in waiting_items],
                "peak_waiting": int(self.peak_waiting),
                "total_submitted": int(self.total_submitted),
                self.completion_key: int(self.total_completed),
            }
        if extra:
            snapshot.update(dict(extra))
        return snapshot

    def peek_oldest_age_ms(self, timestamp_attr: str) -> float:
        with self.condition:
            if not self.queue:
                return 0.0
            enqueue_time = float(getattr(self.queue[0], timestamp_attr))
        return max(0.0, (time.perf_counter() - enqueue_time) * 1000.0)

    def is_drained(self) -> bool:
        with self.condition:
            return not self.queue and self.total_submitted == self.total_completed

    def take_finalize_batch(
        self,
        *,
        finalize_mode: str,
        batch_max_items: int,
        batch_wait_s: float,
        use_vocoder: bool,
    ) -> List[SchedulerFinalizeTask]:
        with self.condition:
            if not self.queue:
                return []
            selected_tasks = [self.queue.popleft()]
            if finalize_mode == "sync" or use_vocoder:
                return selected_tasks
            if batch_max_items <= 1:
                return selected_tasks
            first_task = selected_tasks[0]
            oldest_age_s = max(0.0, time.perf_counter() - first_task.enqueued_time)
            if len(self.queue) + 1 < batch_max_items and oldest_age_s < batch_wait_s:
                self.queue.appendleft(first_task)
                return []
            while len(selected_tasks) < batch_max_items:
                if not self.queue:
                    break
                matched_index = None
                for index, task in enumerate(self.queue):
                    if abs(task.enqueued_time - first_task.enqueued_time) < 1.0:
                        matched_index = index
                        break
                if matched_index is None:
                    break
                selected_tasks.append(self.queue[matched_index])
                del self.queue[matched_index]
            return selected_tasks


class EnginePolicyArbiterController:
    def __init__(
        self,
        *,
        policy_config: EnginePolicyConfig,
        arbiter_config: EngineArbiterConfig,
        snapshot_request_registry: Callable[[], Dict[str, Any]],
        get_worker_state: Callable[[], Dict[str, Any]],
        snapshot_prepare_state: Callable[[], Dict[str, Any]],
        snapshot_finalize_state: Callable[[], Dict[str, Any]],
        snapshot_dispatch_state: Callable[[], Dict[str, Any]],
        snapshot_decode_runtime_state: Callable[[], Dict[str, Any]],
        snapshot_job_registry: Callable[[], Dict[str, Any]],
        peek_queue_age_ms: Callable[[str], float],
        merge_request_state_profile: Callable[[str, Optional[Dict[str, Any]]], None],
    ) -> None:
        self.policy_config = policy_config
        self.policy_poll_s = max(0.001, float(self.policy_config.poll_wait_ms) / 1000.0)
        self.arbiter_config = arbiter_config
        self.arbiter_poll_s = max(0.001, float(self.arbiter_config.poll_wait_ms) / 1000.0)
        self.condition = threading.Condition()
        self.state = EngineArbiterState(
            decode_budget_remaining=int(self.arbiter_config.decode_burst),
            last_observed_at=time.perf_counter(),
        )
        self.snapshot_request_registry = snapshot_request_registry
        self.get_worker_state = get_worker_state
        self.snapshot_prepare_state = snapshot_prepare_state
        self.snapshot_finalize_state = snapshot_finalize_state
        self.snapshot_dispatch_state = snapshot_dispatch_state
        self.snapshot_decode_runtime_state = snapshot_decode_runtime_state
        self.snapshot_job_registry = snapshot_job_registry
        self.peek_queue_age_ms = peek_queue_age_ms
        self.merge_request_state_profile = merge_request_state_profile

    def snapshot_state(self) -> Dict[str, Any]:
        with self.condition:
            return {
                "config": self.arbiter_config.to_dict(),
                "total_ticks": int(self.state.total_ticks),
                "total_idle_ticks": int(self.state.total_idle_ticks),
                "total_prepare_dispatches": int(self.state.total_prepare_dispatches),
                "total_decode_dispatches": int(self.state.total_decode_dispatches),
                "total_decode_runtime_ticks": int(self.state.total_decode_runtime_ticks),
                "total_finalize_dispatches": int(self.state.total_finalize_dispatches),
                "decode_budget_remaining": int(self.state.decode_budget_remaining),
                "last_stage": str(self.state.last_stage),
                "last_reason": str(self.state.last_reason),
                "last_policy_allowed": bool(self.state.last_policy_allowed),
                "last_observed_at": float(self.state.last_observed_at),
            }

    def notify(self) -> None:
        with self.condition:
            self.condition.notify_all()

    def wait(self) -> None:
        with self.condition:
            self.condition.wait(timeout=self.arbiter_poll_s)

    def mark_tick(self, *, stage: str, reason: str, policy_allowed: bool) -> None:
        with self.condition:
            self.state.total_ticks += 1
            if stage == "idle":
                self.state.total_idle_ticks += 1
            elif stage == "prepare":
                self.state.total_prepare_dispatches += 1
                self.state.decode_budget_remaining = int(self.arbiter_config.decode_burst)
            elif stage == "finalize":
                self.state.total_finalize_dispatches += 1
                self.state.decode_budget_remaining = int(self.arbiter_config.decode_burst)
            elif stage == "decode_dispatch":
                self.state.total_decode_dispatches += 1
            elif stage == "decode_runtime":
                self.state.total_decode_runtime_ticks += 1
                self.state.decode_budget_remaining = max(0, int(self.state.decode_budget_remaining) - 1)
            self.state.last_stage = str(stage)
            self.state.last_reason = str(reason)
            self.state.last_policy_allowed = bool(policy_allowed)
            self.state.last_observed_at = time.perf_counter()

    def build_stage_counters(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        prepare_dispatcher_state = self.snapshot_prepare_state()
        finalize_dispatcher_state = self.snapshot_finalize_state()
        dispatcher_state = self.snapshot_dispatch_state()
        active_requests = list(request_registry.get("active_requests", []))
        status_counts: Dict[str, int] = {}
        for item in active_requests:
            status = str(item.get("status", "UNKNOWN"))
            status_counts[status] = status_counts.get(status, 0) + 1

        worker_pending_jobs = int(worker_state.get("pending_jobs", 0))
        worker_decode_active_size = int(worker_state.get("running_requests", 0))
        worker_prepare_inflight = int(worker_state.get("prepare_inflight", 0))
        worker_finalize_pending = int(worker_state.get("finalize_pending", 0))
        worker_finalize_inflight = int(worker_state.get("finalize_inflight", 0))
        engine_decode_runtime_state = self.snapshot_decode_runtime_state()
        engine_job_registry = self.snapshot_job_registry()
        decode_runtime_pending_jobs = int(engine_decode_runtime_state.get("pending_jobs", 0))
        decode_runtime_active_size = int(engine_decode_runtime_state.get("active_request_count", 0))
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
            "worker_pending_jobs": worker_pending_jobs,
            "worker_decode_active_size": worker_decode_active_size,
            "worker_decode_control_enabled": bool(worker_state.get("engine_decode_control_enabled", False)),
            "worker_decode_runtime_has_work": bool(worker_state.get("decode_runtime_has_work", False)),
            "engine_decode_runtime_pending_jobs": decode_runtime_pending_jobs,
            "engine_decode_runtime_active_request_count": decode_runtime_active_size,
            "engine_decode_runtime_has_work": bool(engine_decode_runtime_state.get("has_work", False)),
            "engine_job_registry_count": int(engine_job_registry.get("job_count", 0)),
            "worker_prepare_inflight": worker_prepare_inflight,
            "worker_finalize_pending": worker_finalize_pending,
            "worker_finalize_inflight": worker_finalize_inflight,
            "engine_gpu_prepare_queue_count": int(prepare_dispatcher_state.get("waiting_count", 0)),
            "engine_finalize_queue_count": int(finalize_dispatcher_state.get("waiting_count", 0)),
            "engine_decode_waiting_queue_count": int(dispatcher_state.get("waiting_count", 0)),
            "decode_backlog": int(
                decode_runtime_pending_jobs + decode_runtime_active_size
                if bool(worker_state.get("engine_decode_control_enabled", False))
                else worker_pending_jobs + worker_decode_active_size
            ),
        }

    def build_policy_snapshot(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        counters = self.build_stage_counters(request_registry, worker_state)
        config = self.policy_config.to_dict()
        blocked_reasons: List[Dict[str, Any]] = []
        finalize_pending_total = int(counters["worker_finalize_pending"]) + int(counters.get("engine_finalize_queue_count", 0))
        limit_checks = [
            ("decode_backlog", counters["decode_backlog"], int(config["decode_backlog_soft_max"])),
            ("finalize_pending", finalize_pending_total, int(config["finalize_pending_soft_max"])),
            ("prepare_inflight", counters["worker_prepare_inflight"], int(config["prepare_inflight_soft_max"])),
            ("active_decode_requests", counters["active_decode_request_count"], int(config["active_decode_soft_max"])),
            ("ready_for_prefill_requests", counters["ready_for_prefill_request_count"], int(config["ready_for_prefill_soft_max"])),
            ("active_requests", counters["active_request_count"], int(config["active_request_soft_max"])),
        ]
        if bool(config["enabled"]):
            for name, value, limit in limit_checks:
                if limit > 0 and int(value) >= int(limit):
                    blocked_reasons.append({"metric": name, "value": int(value), "limit": int(limit)})
        return {
            "enabled": bool(config["enabled"]),
            "allowed": (not bool(config["enabled"])) or not blocked_reasons,
            "blocked_reasons": blocked_reasons,
            "config": config,
            "metrics": {
                "active_request_count": int(counters["active_request_count"]),
                "queued_request_count": int(counters["queued_request_count"]),
                "ready_for_prefill_request_count": int(counters["ready_for_prefill_request_count"]),
                "active_decode_request_count": int(counters["active_decode_request_count"]),
                "engine_gpu_prepare_queue_count": int(counters["engine_gpu_prepare_queue_count"]),
                "engine_decode_waiting_queue_count": int(counters["engine_decode_waiting_queue_count"]),
                "decode_backlog": int(counters["decode_backlog"]),
                "prepare_inflight": int(counters["worker_prepare_inflight"]),
                "finalize_pending": int(finalize_pending_total),
                "engine_finalize_queue_count": int(counters.get("engine_finalize_queue_count", 0)),
                "finalize_inflight": int(counters["worker_finalize_inflight"]),
            },
            "observed_at": time.perf_counter(),
        }

    async def wait_for_policy_admission(
        self,
        *,
        request_id: str | None,
        timeout_sec: float | None,
    ) -> tuple[float, Dict[str, Any]]:
        request_registry = self.snapshot_request_registry()
        worker_state = self.get_worker_state()
        snapshot = self.build_policy_snapshot(request_registry, worker_state)
        if not self.policy_config.enabled:
            return 0.0, snapshot
        start = time.perf_counter()
        deadline = None if timeout_sec in [None, ""] else (start + max(0.0, float(timeout_sec)))
        while True:
            request_registry = self.snapshot_request_registry()
            worker_state = self.get_worker_state()
            snapshot = self.build_policy_snapshot(request_registry, worker_state)
            if snapshot["allowed"]:
                wait_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
                if request_id not in [None, ""]:
                    self.merge_request_state_profile(
                        str(request_id),
                        {
                            "engine_policy_wait_ms": float(wait_ms),
                            "engine_policy_snapshot": snapshot,
                        },
                    )
                return wait_ms, snapshot
            now = time.perf_counter()
            if deadline is not None and now >= deadline:
                blocked_summary = ", ".join(
                    f"{item['metric']}={item['value']}/{item['limit']}" for item in snapshot.get("blocked_reasons", [])
                )
                raise TimeoutError(f"engine policy admission timeout ({blocked_summary})")
            await asyncio.sleep(self.policy_poll_s)

    def select_stage(self) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        request_registry = self.snapshot_request_registry()
        worker_state = self.get_worker_state()
        policy_snapshot = self.build_policy_snapshot(request_registry, worker_state)
        prepare_waiting = int(self.snapshot_prepare_state().get("waiting_count", 0))
        finalize_waiting = int(self.snapshot_finalize_state().get("waiting_count", 0))
        decode_waiting = int(self.snapshot_dispatch_state().get("waiting_count", 0))
        decode_runtime_state = self.snapshot_decode_runtime_state()
        worker_decode_has_work = bool(decode_runtime_state.get("has_work", False))
        worker_decode_control_enabled = bool(worker_state.get("engine_decode_control_enabled", False))
        worker_pending_jobs = int(decode_runtime_state.get("pending_jobs", 0))
        worker_running_requests = int(decode_runtime_state.get("active_request_count", 0))
        prepare_age_ms = float(self.peek_queue_age_ms("prepare"))
        finalize_age_ms = float(self.peek_queue_age_ms("finalize"))
        decode_runtime_pending_age_ms = float(self.peek_queue_age_ms("decode_runtime_pending"))
        decode_budget_remaining = int(self.snapshot_state().get("decode_budget_remaining", 0))
        policy_allowed = bool(policy_snapshot.get("allowed", True))
        if (
            worker_decode_control_enabled
            and worker_decode_has_work
            and policy_allowed
            and decode_budget_remaining > 0
            and (worker_running_requests > 0 or worker_pending_jobs > 0)
        ):
            return "decode_runtime", "worker_active_batch_progress", policy_snapshot, worker_state
        if (
            worker_decode_control_enabled
            and worker_pending_jobs > 0
            and policy_allowed
            and decode_runtime_pending_age_ms >= float(self.arbiter_config.prepare_aging_ms)
        ):
            return "decode_runtime", "decode_runtime_pending_aging", policy_snapshot, worker_state
        if (
            decode_waiting > 0
            and policy_allowed
            and (not worker_decode_control_enabled or not worker_decode_has_work or worker_pending_jobs <= 0)
        ):
            return "decode_dispatch", "dispatch_prepared_state", policy_snapshot, worker_state
        if finalize_waiting > 0 and (decode_waiting <= 0 or not policy_allowed or decode_budget_remaining <= 0):
            return "finalize", "decode_blocked_or_budget_exhausted", policy_snapshot, worker_state
        if finalize_waiting > 0 and finalize_age_ms >= float(self.arbiter_config.finalize_aging_ms):
            return "finalize", "finalize_aging", policy_snapshot, worker_state
        if prepare_waiting > 0 and (decode_waiting <= 0 or not policy_allowed or decode_budget_remaining <= 0):
            return "prepare", "decode_blocked_or_budget_exhausted", policy_snapshot, worker_state
        if prepare_waiting > 0 and prepare_age_ms >= float(self.arbiter_config.prepare_aging_ms):
            return "prepare", "prepare_aging", policy_snapshot, worker_state
        if worker_decode_control_enabled and worker_decode_has_work and policy_allowed:
            return "decode_runtime", "worker_active_batch_progress_fallback", policy_snapshot, worker_state
        if decode_waiting > 0 and policy_allowed:
            return "decode_dispatch", "decode_priority_fallback", policy_snapshot, worker_state
        if finalize_waiting > 0:
            return "finalize", "finalize_fallback", policy_snapshot, worker_state
        if prepare_waiting > 0:
            return "prepare", "prepare_fallback", policy_snapshot, worker_state
        return "idle", "no_pending_work", policy_snapshot, worker_state


class EngineDecodeRuntimeOwner:
    def __init__(
        self,
        *,
        get_decode_runtime_counters: Callable[[], Dict[str, int]],
        get_micro_batch_wait_s: Callable[[], float],
    ) -> None:
        self.get_decode_runtime_counters = get_decode_runtime_counters
        self.get_micro_batch_wait_s = get_micro_batch_wait_s
        self.condition = threading.Condition()
        self.pending_jobs: Deque[SchedulerPendingJob] = deque()
        self.active_batch: T2SActiveBatch | None = None
        self.state_lock = threading.Lock()
        self.state = EngineDecodeRuntimeState(updated_at=time.perf_counter())

    @staticmethod
    def summarize_active_batch(active_batch: T2SActiveBatch | None) -> Dict[str, Any]:
        if active_batch is None:
            return {}
        decode_step_index_max = 0
        if active_batch.step_indices is not None and active_batch.step_indices.numel() > 0:
            decode_step_index_max = int(active_batch.step_indices.max().item())
        return {
            "request_count": int(len(active_batch.request_ids)),
            "request_ids": list(active_batch.request_ids),
            "prefill_done": bool(active_batch.prefill_done),
            "decode_step_index_max": int(decode_step_index_max),
        }

    def snapshot_pending_queue_state(self) -> Dict[str, Any]:
        with self.condition:
            return {
                "pending_jobs": int(len(self.pending_jobs)),
                "pending_request_ids": [job.request_id for job in list(self.pending_jobs)[:32]],
            }

    def enqueue_pending_job(self, job: SchedulerPendingJob) -> None:
        with self.condition:
            self.pending_jobs.append(job)
            self.condition.notify_all()
        self.refresh_state("engine_decode_pending_enqueue")

    def take_pending_jobs_nonblocking(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        with self.condition:
            if not self.pending_jobs:
                return []
            if wait_for_batch:
                oldest_enqueue_time = float(self.pending_jobs[0].enqueue_time)
                if (time.perf_counter() - oldest_enqueue_time) < self.get_micro_batch_wait_s():
                    return []
            pending_jobs = list(self.pending_jobs)
            self.pending_jobs.clear()
        self.refresh_state("engine_decode_pending_dequeue")
        return pending_jobs

    def pending_age_ms(self) -> float:
        with self.condition:
            if not self.pending_jobs:
                return 0.0
            enqueue_time = float(self.pending_jobs[0].enqueue_time)
        return max(0.0, (time.perf_counter() - enqueue_time) * 1000.0)

    def has_pending_jobs(self) -> bool:
        with self.condition:
            return bool(self.pending_jobs)

    def get_active_batch(self) -> T2SActiveBatch | None:
        return self.active_batch

    def set_active_batch(self, active_batch: T2SActiveBatch | None) -> None:
        self.active_batch = active_batch

    def active_batch_summary(self) -> Dict[str, Any]:
        return self.summarize_active_batch(self.active_batch)

    def refresh_state(self, last_event: str) -> None:
        pending_state = self.snapshot_pending_queue_state()
        active_batch_summary = self.active_batch_summary()
        worker_decode_counters = self.get_decode_runtime_counters()
        with self.state_lock:
            self.state.pending_jobs = int(pending_state.get("pending_jobs", 0))
            self.state.pending_request_ids = list(pending_state.get("pending_request_ids", []))
            self.state.active_request_count = int(active_batch_summary.get("request_count", 0))
            self.state.active_request_ids = list(active_batch_summary.get("request_ids", []))[:32]
            self.state.prefill_done = bool(active_batch_summary.get("prefill_done", False))
            self.state.decode_step_index_max = int(active_batch_summary.get("decode_step_index_max", 0))
            self.state.total_cycles = int(worker_decode_counters.get("total_cycles", 0))
            self.state.prefill_cycles = int(worker_decode_counters.get("prefill_cycles", 0))
            self.state.step_cycles = int(worker_decode_counters.get("step_cycles", 0))
            self.state.has_work = bool(pending_state.get("pending_jobs", 0) or active_batch_summary.get("request_count", 0))
            self.state.last_event = str(last_event)
            self.state.updated_at = float(time.perf_counter())

    def update_from_worker_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if not snapshot:
            return
        pending_state = self.snapshot_pending_queue_state()
        with self.state_lock:
            self.state.pending_jobs = int(pending_state.get("pending_jobs", 0))
            self.state.pending_request_ids = list(pending_state.get("pending_request_ids", []))
            self.state.active_request_count = int(snapshot.get("active_request_count", 0))
            self.state.active_request_ids = list(snapshot.get("active_request_ids", []))[:32]
            self.state.prefill_done = bool(snapshot.get("prefill_done", False))
            self.state.decode_step_index_max = int(snapshot.get("decode_step_index_max", 0))
            self.state.total_cycles = int(snapshot.get("total_cycles", 0))
            self.state.prefill_cycles = int(snapshot.get("prefill_cycles", 0))
            self.state.step_cycles = int(snapshot.get("step_cycles", 0))
            self.state.has_work = bool(
                pending_state.get("pending_jobs", 0)
                or snapshot.get("active_request_count", 0)
                or snapshot.get("has_work", False)
            )
            self.state.last_event = str(snapshot.get("last_event", "unknown"))
            self.state.updated_at = float(snapshot.get("updated_at", time.perf_counter()))

    def snapshot_state(self) -> Dict[str, Any]:
        pending_state = self.snapshot_pending_queue_state()
        active_batch_summary = self.active_batch_summary()
        worker_decode_counters = self.get_decode_runtime_counters()
        with self.state_lock:
            return {
                "pending_jobs": int(pending_state.get("pending_jobs", self.state.pending_jobs)),
                "pending_request_ids": list(pending_state.get("pending_request_ids", self.state.pending_request_ids)),
                "active_request_count": int(active_batch_summary.get("request_count", self.state.active_request_count)),
                "active_request_ids": list(active_batch_summary.get("request_ids", self.state.active_request_ids)),
                "prefill_done": bool(active_batch_summary.get("prefill_done", self.state.prefill_done)),
                "decode_step_index_max": int(
                    active_batch_summary.get("decode_step_index_max", self.state.decode_step_index_max)
                ),
                "total_cycles": int(worker_decode_counters.get("total_cycles", 0)),
                "prefill_cycles": int(worker_decode_counters.get("prefill_cycles", 0)),
                "step_cycles": int(worker_decode_counters.get("step_cycles", 0)),
                "has_work": bool(
                    pending_state.get("pending_jobs", 0)
                    or active_batch_summary.get("request_count", self.state.active_request_count)
                    or self.state.has_work
                ),
                "last_event": str(self.state.last_event),
                "updated_at": float(self.state.updated_at),
            }

@dataclass
class SchedulerFinalizeTask:
    request_id: str
    item: T2SFinishedItem
    enqueued_time: float


@dataclass
class EngineDispatchTask:
    request_id: str
    state: T2SRequestState
    speed_factor: float
    sample_steps: int
    media_type: str
    prepare_wall_ms: float
    prepare_profile_total_ms: float
    done_loop: asyncio.AbstractEventLoop | None
    done_future: asyncio.Future | None
    engine_request_id: str | None
    timeout_sec: float | None
    enqueue_time: float
    worker_job: SchedulerPendingJob | None = None
    engine_policy_wait_ms: float = 0.0
    engine_dispatch_wait_ms: float = 0.0
    engine_policy_snapshot: Dict[str, Any] | None = None
    error: str | None = None


@dataclass
class EngineGpuPrepareTask:
    request_id: str
    cpu_stage: PreparedCpuStage
    done_loop: asyncio.AbstractEventLoop | None
    done_future: asyncio.Future | None
    engine_request_id: str | None
    enqueue_time: float
    queue_wait_ms: float = 0.0
    error: str | None = None


@dataclass
class EngineFinalizeQueueState:
    waiting_count: int
    waiting_request_ids: List[str]
    peak_waiting: int
    total_submitted: int
    total_completed: int


@dataclass
class EngineArbiterState:
    total_ticks: int = 0
    total_idle_ticks: int = 0
    total_prepare_dispatches: int = 0
    total_decode_dispatches: int = 0
    total_decode_runtime_ticks: int = 0
    total_finalize_dispatches: int = 0
    decode_budget_remaining: int = 0
    last_stage: str = "idle"
    last_reason: str = "init"
    last_observed_at: float = 0.0
    last_policy_allowed: bool = True


@dataclass
class EngineDecodeRuntimeState:
    pending_jobs: int = 0
    pending_request_ids: List[str] = field(default_factory=list)
    active_request_count: int = 0
    active_request_ids: List[str] = field(default_factory=list)
    prefill_done: bool = False
    decode_step_index_max: int = 0
    total_cycles: int = 0
    prefill_cycles: int = 0
    step_cycles: int = 0
    has_work: bool = False
    last_event: str = "init"
    updated_at: float = 0.0


@dataclass
class RuntimeStateCallbacks:
    update: Callable[[str, str, Optional[Dict[str, Any]]], None] | None = None
    complete: Callable[[str, Optional[Dict[str, Any]]], None] | None = None
    fail: Callable[[str, str], None] | None = None
    decode_runtime_update: Callable[[Dict[str, Any]], None] | None = None


class WorkerPrepareExecutor:
    def __init__(
        self,
        tts: TTS,
        on_state_change: Callable[[], None] | None = None,
    ) -> None:
        self.coordinator = PrepareCoordinator(tts)
        self.on_state_change = on_state_change

    def _notify_state_change(self) -> None:
        if self.on_state_change is None:
            return
        try:
            self.on_state_change()
        except Exception:
            pass

    def snapshot(self) -> Dict[str, int]:
        return dict(self.coordinator.snapshot())

    def get_max_inflight(self) -> int:
        return int(self.coordinator.snapshot().get("max_inflight", 0))

    def is_idle(self) -> bool:
        return int(self.coordinator.snapshot().get("inflight", 0)) <= 0

    async def prepare_state_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> tuple[T2SRequestState, float, float]:
        try:
            return await self.coordinator.prepare_state_profiled_async(spec, prepare_submit_at)
        finally:
            self._notify_state_change()

    async def prepare_states_batch_async(self, specs: List[SchedulerRequestSpec]) -> List[T2SRequestState]:
        results = await asyncio.gather(
            *[self.prepare_state_profiled_async(spec, time.perf_counter()) for spec in specs]
        )
        return [state for state, _, _ in results]

    async def prepare_cpu_stage_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> PreparedCpuStage:
        try:
            return await self.coordinator.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)
        finally:
            self._notify_state_change()

    async def prepare_gpu_stage_profiled_async(
        self,
        cpu_stage: PreparedCpuStage,
    ) -> tuple[T2SRequestState, float, float]:
        try:
            return await self.coordinator.prepare_gpu_stage_profiled_async(cpu_stage)
        finally:
            self._notify_state_change()


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

    def synthesize_finalize_jobs(
        self,
        jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]],
    ) -> tuple[float, List[tuple[int, np.ndarray]]]:
        if not jobs_and_items:
            return 0.0, []
        self._sync_device()
        synth_start = time.perf_counter()
        if len(jobs_and_items) == 1 or self.tts.configs.use_vocoder:
            job, item = jobs_and_items[0]
            batch_results = [self._synthesize_finished_audio(job, item)]
        else:
            batch_results = self._synthesize_finished_audio_batch(jobs_and_items)
        self._sync_device()
        synth_ms = (time.perf_counter() - synth_start) * 1000.0
        return float(synth_ms), batch_results


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


class UnifiedSchedulerWorker:
    def __init__(
        self,
        tts: TTS,
        max_steps: int = 1500,
        micro_batch_wait_ms: int = 5,
        runtime_callbacks: RuntimeStateCallbacks | None = None,
        external_finalize_submit: Callable[[List[SchedulerFinalizeTask]], None] | None = None,
    ):
        self.tts = tts
        self.max_steps = int(max_steps)
        self.micro_batch_wait_s = float(micro_batch_wait_ms) / 1000.0
        self.runtime_callbacks = runtime_callbacks or RuntimeStateCallbacks()
        self.condition = threading.Condition()
        self.completion_bridge = WorkerCompletionBridge(self.runtime_callbacks)
        self.decode_executor = WorkerDecodeExecutor(tts, max_steps=max_steps)
        self.decode_legacy_shell = WorkerDecodeLegacyShell(self.condition, self.micro_batch_wait_s)
        self.decode_runtime_tracker = WorkerDecodeRuntimeTracker(self.runtime_callbacks)
        self.prepare_executor = WorkerPrepareExecutor(tts, on_state_change=self._notify_worker_state_change)
        self.finalize_executor = WorkerFinalizeExecutor(
            tts,
            on_state_change=self._notify_worker_state_change,
            external_submit=external_finalize_submit,
        )
        self.decode_backlog_max = max(0, int(os.environ.get("GPTSOVITS_ENGINE_DECODE_BACKLOG_MAX", "0")))
        self.finalize_pending_max = max(0, int(os.environ.get("GPTSOVITS_ENGINE_FINALIZE_PENDING_MAX", "0")))
        self.engine_decode_control_enabled = (
            str(os.environ.get("GPTSOVITS_ENGINE_DRIVE_DECODE", "0")).strip().lower() in {"1", "true", "yes", "on"}
        )
        self.job_registry = SchedulerJobRegistry(self.condition)
        self.worker_thread: threading.Thread | None = None
        if not self.engine_decode_control_enabled:
            self.worker_thread = threading.Thread(target=self._run_loop, name="unified-t2s-scheduler-worker", daemon=True)
            self.worker_thread.start()
        self.finalize_threads = []
        if external_finalize_submit is None:
            self.finalize_threads = [
                threading.Thread(
                    target=self._run_finalize_loop,
                    name=f"unified-t2s-finalize-{worker_index}",
                    daemon=True,
                )
                for worker_index in range(self.finalize_executor.get_worker_count())
            ]
            for finalize_thread in self.finalize_threads:
                finalize_thread.start()

    def _notify_worker_state_change(self) -> None:
        with self.condition:
            self.condition.notify_all()

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

    def _admission_snapshot_locked(self) -> Dict[str, int]:
        _, snapshot = self._can_accept_submit_locked()
        return snapshot

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
    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value in [None, ""]:
            return int(default)
        return int(value)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value in [None, ""]:
            return float(default)
        return float(value)

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
        self.request_registry = EngineRequestRegistry(
            recent_limit=max(1, int(os.environ.get("GPTSOVITS_ENGINE_RECENT_REQUEST_LIMIT", "64")))
        )
        self.engine_job_registry = SchedulerJobRegistry(threading.Lock())
        self.scheduler_worker = UnifiedSchedulerWorker(
            tts,
            max_steps=max_steps,
            micro_batch_wait_ms=micro_batch_wait_ms,
            runtime_callbacks=RuntimeStateCallbacks(
                update=self._update_request_state,
                complete=self._complete_request_state,
                fail=self._fail_request_state,
                decode_runtime_update=self._update_engine_decode_runtime_state,
            ),
            external_finalize_submit=self._enqueue_worker_finished_for_finalize,
        )
        self.direct_tts_lock = threading.RLock()
        self.management_lock = threading.RLock()
        worker_capacity_limits = self.scheduler_worker.get_capacity_limits()
        prepare_max_inflight = int(self.scheduler_worker.get_prepare_max_inflight())
        self.engine_policy_config = EnginePolicyConfig(
            enabled=self._env_flag("GPTSOVITS_ENGINE_POLICY_ENABLE", True),
            poll_wait_ms=max(1.0, self._env_float("GPTSOVITS_ENGINE_POLICY_POLL_WAIT_MS", float(micro_batch_wait_ms))),
            decode_backlog_soft_max=max(
                0,
                self._env_int(
                    "GPTSOVITS_ENGINE_POLICY_DECODE_BACKLOG_SOFT_MAX",
                    int(worker_capacity_limits["decode_backlog_max"]),
                ),
            ),
            finalize_pending_soft_max=max(
                0,
                self._env_int(
                    "GPTSOVITS_ENGINE_POLICY_FINALIZE_PENDING_SOFT_MAX",
                    int(worker_capacity_limits["finalize_pending_max"]),
                ),
            ),
            prepare_inflight_soft_max=max(
                0,
                self._env_int("GPTSOVITS_ENGINE_POLICY_PREPARE_INFLIGHT_SOFT_MAX", prepare_max_inflight),
            ),
            active_decode_soft_max=max(0, self._env_int("GPTSOVITS_ENGINE_POLICY_ACTIVE_DECODE_SOFT_MAX", 0)),
            ready_for_prefill_soft_max=max(0, self._env_int("GPTSOVITS_ENGINE_POLICY_READY_FOR_PREFILL_SOFT_MAX", 0)),
            active_request_soft_max=max(0, self._env_int("GPTSOVITS_ENGINE_POLICY_ACTIVE_REQUEST_SOFT_MAX", 0)),
        )
        self.engine_arbiter_config = EngineArbiterConfig(
            poll_wait_ms=max(1.0, self._env_float("GPTSOVITS_ENGINE_ARBITER_POLL_WAIT_MS", float(micro_batch_wait_ms))),
            decode_burst=max(1, self._env_int("GPTSOVITS_ENGINE_ARBITER_DECODE_BURST", 4)),
            prepare_aging_ms=max(0.0, self._env_float("GPTSOVITS_ENGINE_ARBITER_PREPARE_AGING_MS", 10.0)),
            finalize_aging_ms=max(0.0, self._env_float("GPTSOVITS_ENGINE_ARBITER_FINALIZE_AGING_MS", 10.0)),
        )
        self.engine_decode_runtime_owner = EngineDecodeRuntimeOwner(
            get_decode_runtime_counters=self.scheduler_worker.get_decode_runtime_counters,
            get_micro_batch_wait_s=self.scheduler_worker.get_micro_batch_wait_s,
        )
        self.engine_prepare_queue_owner = EngineTaskQueueOwner(completion_key="total_completed")
        self.engine_finalize_queue_owner = EngineTaskQueueOwner(completion_key="total_completed")
        self.engine_dispatch_queue_owner = EngineTaskQueueOwner(completion_key="total_dispatched")
        self.engine_dispatch_last_snapshot: Dict[str, Any] = {}
        self.engine_policy_arbiter = EnginePolicyArbiterController(
            policy_config=self.engine_policy_config,
            arbiter_config=self.engine_arbiter_config,
            snapshot_request_registry=self._snapshot_request_registry,
            get_worker_state=self.get_scheduler_state,
            snapshot_prepare_state=self._snapshot_engine_prepare_state,
            snapshot_finalize_state=self._snapshot_engine_finalize_state,
            snapshot_dispatch_state=self._snapshot_engine_dispatch_state,
            snapshot_decode_runtime_state=self._snapshot_engine_decode_runtime_state,
            snapshot_job_registry=self._snapshot_engine_job_registry,
            peek_queue_age_ms=self._peek_queue_age_ms,
            merge_request_state_profile=self._merge_request_state_profile,
        )
        self.engine_arbiter_thread = threading.Thread(
            target=self._run_engine_arbiter_loop,
            name="unified-engine-arbiter",
            daemon=True,
        )
        self.engine_arbiter_thread.start()

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
        return self.request_registry.register(
            request_id=request_id,
            api_mode=api_mode,
            backend=backend,
            media_type=media_type,
            response_streaming=response_streaming,
            deadline_ts=deadline_ts,
            meta=meta,
        )

    def _update_request_state(
        self,
        request_id: str,
        status: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.request_registry.update(request_id, status, extra)

    def _merge_request_state_profile(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.request_registry.merge_profile(request_id, extra)

    def _snapshot_engine_prepare_state(self) -> Dict[str, Any]:
        return self.engine_prepare_queue_owner.snapshot(max_request_ids=16)

    def _snapshot_engine_finalize_state(self) -> Dict[str, Any]:
        return self.engine_finalize_queue_owner.snapshot(max_request_ids=16)

    def _snapshot_engine_dispatch_state(self) -> Dict[str, Any]:
        return self.engine_dispatch_queue_owner.snapshot(
            max_request_ids=16,
            extra={"last_policy_snapshot": dict(self.engine_dispatch_last_snapshot or {})},
        )

    def _register_engine_job(self, job: SchedulerPendingJob) -> None:
        self.engine_job_registry.register(job, keep_job=True)

    def _get_engine_job(self, request_id: str) -> SchedulerPendingJob | None:
        return self.engine_job_registry.get(request_id)

    def _pop_engine_job(self, request_id: str) -> SchedulerPendingJob | None:
        return self.engine_job_registry.pop(request_id)

    def _snapshot_engine_job_registry(self) -> Dict[str, Any]:
        return self.engine_job_registry.snapshot(max_request_ids=32)

    def _is_engine_drained(self) -> bool:
        prepare_empty = self.engine_prepare_queue_owner.is_drained()
        dispatch_empty = self.engine_dispatch_queue_owner.is_drained()
        finalize_empty = self.engine_finalize_queue_owner.is_drained()
        decode_pending_empty = not self.engine_decode_runtime_owner.has_pending_jobs()
        job_empty = self.engine_job_registry.is_empty()
        worker_state = self.scheduler_worker.snapshot()
        return bool(
            prepare_empty
            and dispatch_empty
            and finalize_empty
            and decode_pending_empty
            and job_empty
            and self.engine_decode_runtime_owner.get_active_batch() is None
            and int(worker_state.get("prepare_inflight", 0)) <= 0
            and int(worker_state.get("finalize_inflight", 0)) <= 0
            and int(worker_state.get("finalize_pending", 0)) <= 0
        )

    def _record_engine_job_done(self, request_id: str) -> None:
        self.engine_job_registry.mark_finished_and_remove(request_id)
        self.scheduler_worker.record_external_job_done(request_id)

    def _complete_engine_job(
        self,
        job: SchedulerPendingJob,
        item: T2SFinishedItem,
        *,
        sample_rate: int,
        audio_data: np.ndarray,
    ) -> None:
        completion_bridge = self.scheduler_worker.completion_bridge
        completion_bridge.build_completed_job_result(job, item, sample_rate=sample_rate, audio_data=audio_data)
        completion_bridge.complete_job(
            job,
            runtime_request_id=job.engine_request_id,
            runtime_extra=completion_bridge.build_runtime_complete_payload(job, item, sample_rate=sample_rate),
            on_job_finished=lambda rid=item.request_id: self._record_engine_job_done(rid),
        )

    def _fail_engine_jobs(self, request_ids: List[str], error: str) -> None:
        if not request_ids:
            return
        completion_bridge = self.scheduler_worker.completion_bridge
        for request_id in request_ids:
            job = self._get_engine_job(request_id)
            if job is None:
                continue
            completion_bridge.fail_job(
                job,
                error=error,
                on_job_finished=lambda rid=request_id: self._record_engine_job_done(rid),
            )

    def _add_engine_prefill_time(self, jobs: List[SchedulerPendingJob], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        for job in jobs:
            job.prefill_ms += delta_ms

    def _add_engine_merge_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        for request_id in request_ids:
            job = self._get_engine_job(request_id)
            if job is not None:
                job.merge_ms += delta_ms

    def _add_engine_decode_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        activate_request_ids: List[str] = []
        for request_id in request_ids:
            job = self._get_engine_job(request_id)
            if job is None:
                continue
            if job.decode_steps == 0:
                activate_request_ids.append(job.engine_request_id)
            job.decode_ms += delta_ms
            job.decode_steps += 1
        for engine_request_id in activate_request_ids:
            self._update_request_state(engine_request_id, EngineStatus.ACTIVE_DECODE, None)

    def _enqueue_engine_finished_items(self, items: List[T2SFinishedItem]) -> None:
        if not items:
            return
        enqueued_at = time.perf_counter()
        tasks = [SchedulerFinalizeTask(request_id=item.request_id, item=item, enqueued_time=enqueued_at) for item in items]
        self._enqueue_worker_finished_for_finalize(tasks)

    def _snapshot_engine_decode_pending_queue_state(self) -> Dict[str, Any]:
        return self.engine_decode_runtime_owner.snapshot_pending_queue_state()

    @staticmethod
    def _summarize_active_batch(active_batch: T2SActiveBatch | None) -> Dict[str, Any]:
        return EngineDecodeRuntimeOwner.summarize_active_batch(active_batch)

    def _refresh_engine_decode_runtime_state(self, last_event: str) -> None:
        self.engine_decode_runtime_owner.refresh_state(last_event)

    def _update_engine_decode_runtime_state(self, snapshot: Dict[str, Any]) -> None:
        if not snapshot:
            return
        if self.scheduler_worker.is_engine_decode_control_enabled():
            return
        self.engine_decode_runtime_owner.update_from_worker_snapshot(snapshot)

    def _snapshot_engine_decode_runtime_state(self) -> Dict[str, Any]:
        return self.engine_decode_runtime_owner.snapshot_state()

    def _snapshot_engine_arbiter_state(self) -> Dict[str, Any]:
        return self.engine_policy_arbiter.snapshot_state()

    def _notify_engine_arbiter(self) -> None:
        self.engine_policy_arbiter.notify()

    def _enqueue_engine_decode_pending_job(self, job: SchedulerPendingJob) -> None:
        self.engine_decode_runtime_owner.enqueue_pending_job(job)
        self._notify_engine_arbiter()

    def _take_engine_decode_pending_jobs_nonblocking(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        return self.engine_decode_runtime_owner.take_pending_jobs_nonblocking(wait_for_batch)

    def _peek_queue_age_ms(self, queue_name: str) -> float:
        if queue_name == "prepare":
            return self.engine_prepare_queue_owner.peek_oldest_age_ms("enqueue_time")
        elif queue_name == "finalize":
            return self.engine_finalize_queue_owner.peek_oldest_age_ms("enqueued_time")
        elif queue_name == "decode_runtime_pending":
            return self.engine_decode_runtime_owner.pending_age_ms()
        else:
            return self.engine_dispatch_queue_owner.peek_oldest_age_ms("enqueue_time")

    def _engine_has_pending_work(self) -> bool:
        if self.scheduler_worker.is_engine_decode_control_enabled():
            if self.engine_decode_runtime_owner.has_pending_jobs():
                return True
        if self.scheduler_worker.is_engine_decode_control_enabled() and self._snapshot_engine_decode_runtime_state().get("active_request_count", 0) > 0:
            return True
        if self.engine_prepare_queue_owner.has_items():
            return True
        if self.engine_finalize_queue_owner.has_items():
            return True
        return self.engine_dispatch_queue_owner.has_items()

    @staticmethod
    def _resolve_dispatch_error_future(future: asyncio.Future, error: Exception) -> None:
        if future.done():
            return
        future.set_exception(error)

    def _notify_dispatch_error(self, task: EngineDispatchTask, error: Exception) -> None:
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_dispatch_error_future, task.done_future, error)
        except RuntimeError:
            pass

    @staticmethod
    def _resolve_prepare_future(
        future: asyncio.Future,
        payload: tuple[T2SRequestState, float, float],
    ) -> None:
        if future.done():
            return
        future.set_result(payload)

    def _notify_prepare_error(self, task: EngineGpuPrepareTask, error: Exception) -> None:
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_dispatch_error_future, task.done_future, error)
        except RuntimeError:
            pass

    def _notify_prepare_result(
        self,
        task: EngineGpuPrepareTask,
        payload: tuple[T2SRequestState, float, float],
    ) -> None:
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_prepare_future, task.done_future, payload)
        except RuntimeError:
            pass

    async def _prepare_state_via_engine_gpu_queue(
        self,
        *,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
        engine_request_id: str | None,
    ) -> tuple[T2SRequestState, float, float]:
        cpu_stage = await self.scheduler_worker.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)
        if engine_request_id not in [None, ""]:
            self._update_request_state(
                str(engine_request_id),
                EngineStatus.GPU_PREPARING,
                {
                    "prompt_text_cpu_queue_ms": float(cpu_stage.prompt_cpu_profiled.queue_ms),
                    "prompt_text_cpu_run_ms": float(cpu_stage.prompt_cpu_profiled.run_ms),
                    "text_cpu_queue_ms": float(cpu_stage.target_cpu_profiled.queue_ms),
                    "text_cpu_run_ms": float(cpu_stage.target_cpu_profiled.run_ms),
                },
            )
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        task = EngineGpuPrepareTask(
            request_id=spec.request_id,
            cpu_stage=cpu_stage,
            done_loop=loop,
            done_future=done_future,
            engine_request_id=engine_request_id or spec.request_id,
            enqueue_time=time.perf_counter(),
        )
        self.engine_prepare_queue_owner.enqueue(task)
        self._notify_engine_arbiter()
        state, prepare_exec_started_at, prepare_exec_finished_at = await done_future
        return state, prepare_exec_started_at, prepare_exec_finished_at

    def _enqueue_worker_finished_for_finalize(self, tasks: List[SchedulerFinalizeTask]) -> None:
        if not tasks:
            return
        for task in tasks:
            job = self._get_engine_job(task.request_id)
            if job is not None:
                self._update_request_state(
                    job.engine_request_id,
                    EngineStatus.READY_FOR_FINALIZE,
                    {
                        "finish_reason": task.item.finish_reason,
                        "semantic_len": int(task.item.semantic_tokens.shape[0]),
                        "finish_idx": int(task.item.finish_idx),
                    },
                )
        self.engine_finalize_queue_owner.enqueue_many(tasks)
        self._notify_engine_arbiter()

    def _take_engine_finalize_batch_nonblocking(self) -> List[SchedulerFinalizeTask]:
        finalize_policy = self.scheduler_worker.get_finalize_batch_policy()
        return self.engine_finalize_queue_owner.take_finalize_batch(
            finalize_mode=str(finalize_policy.get("finalize_mode", "async")),
            batch_max_items=int(finalize_policy.get("finalize_batch_max_items", 1)),
            batch_wait_s=float(finalize_policy.get("finalize_batch_wait_s", 0.0)),
            use_vocoder=bool(self.tts.configs.use_vocoder),
        )

    async def _enqueue_prepared_state_for_dispatch(
        self,
        *,
        state: T2SRequestState,
        speed_factor: float,
        sample_steps: int,
        media_type: str,
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
            prepare_wall_ms=float(prepare_wall_ms),
            prepare_profile_total_ms=float(prepare_profile_total_ms),
            done_loop=done_loop,
            done_future=done_future,
            engine_request_id=engine_request_id or state.request_id,
            timeout_sec=timeout_sec,
            enqueue_time=time.perf_counter(),
        )
        self.engine_dispatch_queue_owner.enqueue(task)
        self._notify_engine_arbiter()
        self._merge_request_state_profile(
            task.engine_request_id or task.request_id,
            {
                "engine_dispatch_queue_depth_on_enqueue": int(self._snapshot_engine_dispatch_state()["waiting_count"]),
            },
        )
        return task

    def _mark_arbiter_tick(self, *, stage: str, reason: str, policy_allowed: bool) -> None:
        self.engine_policy_arbiter.mark_tick(stage=stage, reason=reason, policy_allowed=policy_allowed)

    def _select_engine_stage(self) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        stage, reason, policy_snapshot, worker_state = self.engine_policy_arbiter.select_stage()
        self.engine_dispatch_last_snapshot = dict(policy_snapshot)
        return stage, reason, policy_snapshot, worker_state

    def _run_engine_prepare_once(self) -> bool:
        task = self.engine_prepare_queue_owner.pop_left()
        if task is None:
            return False
        queue_wait_ms = max(0.0, (time.perf_counter() - task.enqueue_time) * 1000.0)
        try:
            state, prepare_exec_started_at, prepare_exec_finished_at = asyncio.run(
                self.scheduler_worker.prepare_gpu_stage_profiled_async(task.cpu_stage)
            )
            state.prepare_profile["engine_gpu_prepare_queue_wait_ms"] = float(queue_wait_ms)
            if task.engine_request_id not in [None, ""]:
                self._merge_request_state_profile(
                    str(task.engine_request_id),
                    {"engine_gpu_prepare_queue_wait_ms": float(queue_wait_ms)},
                )
            self.engine_prepare_queue_owner.mark_completed(1)
            self._notify_prepare_result(task, (state, prepare_exec_started_at, prepare_exec_finished_at))
            return True
        except Exception as exc:
            task.error = str(exc)
            self._fail_request_state(task.engine_request_id or task.request_id, str(exc))
            self._notify_prepare_error(task, exc)
            return True

    def _run_engine_finalize_once(self) -> bool:
        tasks = self._take_engine_finalize_batch_nonblocking()
        if not tasks:
            return False
        self.scheduler_worker.begin_finalize_execution(len(tasks))
        try:
            jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]] = []
            for task in tasks:
                job = self._get_engine_job(task.request_id)
                if job is None:
                    continue
                jobs_and_items.append((job, task.item))
            if not jobs_and_items:
                return False
            now = time.perf_counter()
            for task in tasks:
                job = self._get_engine_job(task.request_id)
                if job is not None:
                    job.finalize_wait_ms += max(0.0, (now - task.enqueued_time) * 1000.0)
            for job, item in jobs_and_items:
                self._update_request_state(
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
                self._complete_engine_job(job, item, sample_rate=sample_rate, audio_data=audio_data)
        except Exception as exc:
            self._fail_engine_jobs([task.request_id for task in tasks], str(exc))
        finally:
            self.scheduler_worker.end_finalize_execution(len(tasks))
        self.engine_finalize_queue_owner.mark_completed(len(tasks), notify=True)
        return True

    def _run_engine_dispatch_once(self, policy_snapshot: Dict[str, Any], worker_state: Dict[str, Any]) -> bool:
        if not bool(policy_snapshot.get("allowed", True)):
            return False
        dispatch_task = self.engine_dispatch_queue_owner.pop_left()
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
            self._register_engine_job(worker_job)
            if self.scheduler_worker.is_engine_decode_control_enabled():
                self._enqueue_engine_decode_pending_job(worker_job)
            self.engine_dispatch_queue_owner.mark_completed(1)
            return True
        except Exception as exc:
            dispatch_task.error = str(exc)
            self._fail_request_state(dispatch_task.engine_request_id or dispatch_task.request_id, str(exc))
            self._notify_dispatch_error(dispatch_task, exc)
            return True

    def _run_engine_decode_runtime_once(self) -> bool:
        if not self.scheduler_worker.is_engine_decode_control_enabled():
            return False
        runtime_state = self._snapshot_engine_decode_runtime_state()
        pending_jobs = self._take_engine_decode_pending_jobs_nonblocking(
            wait_for_batch=int(runtime_state.get("active_request_count", 0)) <= 0
        )
        result = self.scheduler_worker.execute_decode_cycle(
            pending_jobs=pending_jobs,
            active_batch=self.engine_decode_runtime_owner.get_active_batch(),
            external_bookkeeping=True,
        )
        prefill_phase = dict(result.get("prefill_phase") or {})
        if prefill_phase.get("error"):
            self._fail_engine_jobs(list(prefill_phase.get("error_request_ids") or []), str(prefill_phase.get("error")))
        else:
            prefill_jobs = list(prefill_phase.get("pending_jobs") or [])
            self._add_engine_prefill_time(prefill_jobs, float(prefill_phase.get("prefill_elapsed_s", 0.0)))
            self._add_engine_merge_time(
                [] if result.get("active_batch") is None else list(result["active_batch"].request_ids),
                float(prefill_phase.get("merge_elapsed_s", 0.0)),
            )
            self._enqueue_engine_finished_items(list(prefill_phase.get("finished_items") or []))
        decode_phase = dict(result.get("decode_phase") or {})
        if decode_phase.get("error"):
            self._fail_engine_jobs(list(decode_phase.get("error_request_ids") or []), str(decode_phase.get("error")))
        else:
            self._add_engine_decode_time(
                list(decode_phase.get("request_ids") or []),
                float(decode_phase.get("decode_elapsed_s", 0.0)),
            )
            self._enqueue_engine_finished_items(list(decode_phase.get("finished_items") or []))
        self.engine_decode_runtime_owner.set_active_batch(result.get("active_batch"))
        if result.get("executed", False):
            self._refresh_engine_decode_runtime_state("engine_decode_cycle")
        return bool(result.get("executed", False))

    def _run_engine_arbiter_loop(self) -> None:
        while True:
            if not self._engine_has_pending_work():
                self._mark_arbiter_tick(stage="idle", reason="no_pending_work", policy_allowed=True)
                self.engine_policy_arbiter.wait()
                continue
            stage, reason, policy_snapshot, worker_state = self._select_engine_stage()
            policy_allowed = bool(policy_snapshot.get("allowed", True))
            executed = False
            if stage == "prepare":
                executed = self._run_engine_prepare_once()
            elif stage == "finalize":
                executed = self._run_engine_finalize_once()
            elif stage == "decode_dispatch":
                executed = self._run_engine_dispatch_once(policy_snapshot, worker_state)
            elif stage == "decode_runtime":
                executed = self._run_engine_decode_runtime_once()
            if not executed:
                self._mark_arbiter_tick(stage="idle", reason=f"{stage}_not_ready", policy_allowed=policy_allowed)
                self.engine_policy_arbiter.wait()
                continue
            self._mark_arbiter_tick(stage=stage, reason=reason, policy_allowed=policy_allowed)

    def _complete_request_state(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.request_registry.complete(request_id, extra)

    def _fail_request_state(self, request_id: str, error: str) -> None:
        self.request_registry.fail(request_id, error)

    def _snapshot_request_registry(self) -> Dict[str, Any]:
        return self.request_registry.snapshot()

    @staticmethod
    def _safe_component_snapshot(component: Any) -> Dict[str, Any] | None:
        if component is None or not hasattr(component, "snapshot"):
            return None
        try:
            return dict(component.snapshot())
        except Exception:
            return None

    def _build_stage_counters(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.engine_policy_arbiter.build_stage_counters(request_registry, worker_state)

    def _build_engine_policy_snapshot(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.engine_policy_arbiter.build_policy_snapshot(request_registry, worker_state)

    async def _wait_for_engine_policy_admission(
        self,
        *,
        request_id: str | None,
        timeout_sec: float | None,
    ) -> tuple[float, Dict[str, Any]]:
        return await self.engine_policy_arbiter.wait_for_policy_admission(
            request_id=request_id,
            timeout_sec=timeout_sec,
        )

    def _build_stage_summary(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        counters = self._build_stage_counters(request_registry, worker_state)
        bert_worker_state = self._safe_component_snapshot(getattr(self.tts, "prepare_bert_batch_worker", None))
        ref_semantic_worker_state = self._safe_component_snapshot(getattr(self.tts, "prepare_ref_semantic_batch_worker", None))
        text_preprocessor_state = self._safe_component_snapshot(getattr(self.tts, "text_preprocessor", None))

        return {
            **counters,
            "engine_drained": bool(self._is_engine_drained()),
            "admission_config": {
                "decode_backlog_max": int(worker_state.get("decode_backlog_max", 0)),
                "finalize_pending_max": int(worker_state.get("finalize_pending_max", 0)),
            },
            "engine_policy": self._build_engine_policy_snapshot(request_registry, worker_state),
            "engine_arbiter_state": self._snapshot_engine_arbiter_state(),
            "engine_decode_runtime_state": self._snapshot_engine_decode_runtime_state(),
            "engine_job_registry": self._snapshot_engine_job_registry(),
            "engine_active_batch_state": self.engine_decode_runtime_owner.active_batch_summary(),
            "engine_prepare_state": self._snapshot_engine_prepare_state(),
            "engine_finalize_state": self._snapshot_engine_finalize_state(),
            "engine_dispatcher_state": self._snapshot_engine_dispatch_state(),
            "active_batch": dict(worker_state.get("active_batch") or {}),
            "prepare_state": dict(worker_state.get("prepare_state") or {}),
            "bert_batch_worker_state": bert_worker_state,
            "ref_semantic_worker_state": ref_semantic_worker_state,
            "text_preprocessor_state": text_preprocessor_state,
        }

    def _collect_request_summaries(self, request_ids: Sequence[str]) -> List[Dict[str, Any]]:
        return self.request_registry.collect_summaries(request_ids)

    def _has_active_request(self, request_id: str) -> bool:
        return self.request_registry.has_active(request_id)

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
                    "prepare_engine_gpu_queue_wait_ms": float(
                        dict(prepare_item.get("prepare_profile", {})).get("engine_gpu_prepare_queue_wait_ms", 0.0)
                    ),
                    "engine_policy_wait_ms": float(prepare_item.get("engine_policy_wait_ms", 0.0)),
                    "engine_dispatch_wait_ms": float(prepare_item.get("engine_dispatch_wait_ms", 0.0)),
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
        engine_policy_wait_ms = self._sum_profile_field(prepare_profiles, "engine_policy_wait_ms")
        engine_dispatch_wait_ms = self._sum_profile_field(prepare_profiles, "engine_dispatch_wait_ms")
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
            request_total_ms - prepare_wall_ms - engine_policy_wait_ms - worker_total_ms - pack_ms - response_overhead_ms,
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
            "engine_policy_wait_ms": engine_policy_wait_ms,
            "engine_dispatch_wait_ms": engine_dispatch_wait_ms,
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
        engine_policy_wait_ms: float,
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
            request_total_ms
            - prepare_wall_ms
            - engine_policy_wait_ms
            - api_after_prepare_ms
            - worker_total_ms
            - api_wait_result_ms
            - pack_ms,
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
            "engine_policy_wait_ms": float(engine_policy_wait_ms),
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
            "X-Engine-Policy-Wait-Ms": self._format_ms_header(profile.get("engine_policy_wait_ms", 0.0)),
            "X-Engine-Dispatch-Wait-Ms": self._format_ms_header(profile.get("engine_dispatch_wait_ms", 0.0)),
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
                "X-Prepare-Engine-GPU-Queue-Wait-Ms": self._format_ms_header(prepare_profile.get("engine_gpu_prepare_queue_wait_ms", 0.0)),
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
                self._prepare_state_via_engine_gpu_queue(
                    spec=spec,
                    prepare_submit_at=time.perf_counter(),
                    engine_request_id=None,
                )
                for spec in segment_specs
            ]
        )
        prepare_profiles: List[Dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        done_futures: List[asyncio.Future] = []
        self._update_request_state(
            request_id,
            EngineStatus.READY_FOR_PREFILL,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct", "segment_count": len(segment_specs)},
        )
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
            await self._enqueue_prepared_state_for_dispatch(
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
        self._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct"},
        )
        timeout_sec = float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0)
        jobs: List[SchedulerPendingJob] = list(await asyncio.wait_for(asyncio.gather(*done_futures), timeout=timeout_sec))
        for profile_item, job in zip(prepare_profiles, jobs):
            profile_item["engine_policy_wait_ms"] = float(job.engine_policy_wait_ms)
            profile_item["engine_dispatch_wait_ms"] = float(job.engine_dispatch_wait_ms)
        self._merge_request_state_profile(
            request_id,
            {
                "engine_policy_wait_ms": sum(float(job.engine_policy_wait_ms) for job in jobs),
                "engine_dispatch_wait_ms": sum(float(job.engine_dispatch_wait_ms) for job in jobs),
                "prepare_aggregate": self._aggregate_numeric_dicts(
                    [item["prepare_profile"] for item in prepare_profiles]
                ),
            },
        )

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
            state, prepare_exec_started_at, prepare_exec_finished_at = await self._prepare_state_via_engine_gpu_queue(
                spec=spec,
                prepare_submit_at=spec_ready_at,
                engine_request_id=spec.request_id,
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
        await self._enqueue_prepared_state_for_dispatch(
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
            job = await asyncio.wait_for(done_future, timeout=float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0))
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
            engine_policy_wait_ms=float(job.result.get("engine_policy_wait_ms", 0.0)),
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
        engine_policy = self._build_engine_policy_snapshot(request_registry, scheduler_state)
        engine_arbiter_state = self._snapshot_engine_arbiter_state()
        engine_decode_runtime_state = self._snapshot_engine_decode_runtime_state()
        engine_job_registry = self._snapshot_engine_job_registry()
        engine_prepare_state = self._snapshot_engine_prepare_state()
        engine_finalize_state = self._snapshot_engine_finalize_state()
        engine_dispatcher_state = self._snapshot_engine_dispatch_state()
        engine_drained = self._is_engine_drained()
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
            "engine_policy": engine_policy,
            "engine_arbiter_state": engine_arbiter_state,
            "engine_decode_runtime_state": engine_decode_runtime_state,
            "engine_job_registry": engine_job_registry,
            "engine_active_batch_state": self.engine_decode_runtime_owner.active_batch_summary(),
            "engine_prepare_state": engine_prepare_state,
            "engine_finalize_state": engine_finalize_state,
            "engine_dispatcher_state": engine_dispatcher_state,
            "engine_drained": bool(engine_drained),
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
