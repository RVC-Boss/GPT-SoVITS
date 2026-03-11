from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SActiveBatch, T2SFinishedItem, T2SRequestState


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


