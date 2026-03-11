from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import PreparedCpuStage
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SActiveBatch, T2SFinishedItem, T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_component_registry import SchedulerPendingJob


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
                "decode_step_index_max": int(active_batch_summary.get("decode_step_index_max", self.state.decode_step_index_max)),
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
class RuntimeStateCallbacks:
    update: Callable[[str, str, Optional[Dict[str, Any]]], None] | None = None
    complete: Callable[[str, Optional[Dict[str, Any]]], None] | None = None
    fail: Callable[[str, str], None] | None = None
    decode_runtime_update: Callable[[Dict[str, Any]], None] | None = None
