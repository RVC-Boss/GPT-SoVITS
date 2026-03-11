from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Sequence

import numpy as np

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SRequestState


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

    def collect_summaries(self, request_ids: Sequence[str]) -> list[Dict[str, Any]]:
        requested = set(request_ids)
        results: list[Dict[str, Any]] = []
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
