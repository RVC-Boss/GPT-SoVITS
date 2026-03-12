import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Tuple


@dataclass
class TextCpuTask:
    text: str
    language: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.perf_counter)
    enqueued_at: float = 0.0
    admission_wait_ms: float = 0.0
    backpressure_wait_ms: float = 0.0
    capacity_wait_ms: float = 0.0
    pending_depth_on_enqueue: int = 0
    done_event: threading.Event = field(default_factory=threading.Event)
    done_loop: asyncio.AbstractEventLoop | None = None
    done_future: asyncio.Future | None = None
    result: Any = None
    error: Exception | None = None
    profile: Dict[str, float] = field(default_factory=dict)


class PrepareTextCpuWorker:
    def __init__(
        self,
        process_fn: Callable[[str, str], Any],
        worker_count: int,
        max_pending_tasks: int = 0,
        admission_poll_ms: int = 1,
        admission_controller: Callable[[], Dict[str, float | int | bool]] | None = None,
    ) -> None:
        self.process_fn = process_fn
        self.worker_count = max(1, int(worker_count))
        self.max_pending_tasks = max(0, int(max_pending_tasks))
        self.admission_poll_s = max(0.0005, float(max(1, int(admission_poll_ms))) / 1000.0)
        self.admission_controller = admission_controller

        self.condition = threading.Condition()
        self.pending_tasks: Deque[TextCpuTask] = deque()
        self.pending_peak = 0
        self.total_submitted = 0
        self.total_finished = 0
        self.active_workers = 0
        self.active_workers_peak = 0
        self.admission_wait_total_ms = 0.0
        self.admission_wait_peak_ms = 0.0
        self.backpressure_wait_total_ms = 0.0
        self.backpressure_wait_peak_ms = 0.0
        self.capacity_wait_total_ms = 0.0
        self.capacity_wait_peak_ms = 0.0
        self.backpressure_blocked_total = 0

        self.worker_threads = [
            threading.Thread(target=self._run_loop, name=f"prepare-text-cpu-worker-{index}", daemon=True)
            for index in range(self.worker_count)
        ]
        for thread in self.worker_threads:
            thread.start()

    def _can_enqueue_locked(self) -> bool:
        if self.max_pending_tasks <= 0:
            return True
        return (len(self.pending_tasks) + self.active_workers) < self.max_pending_tasks

    def _get_admission_state(self) -> Dict[str, float | int | bool]:
        if self.admission_controller is None:
            return {"blocked": False}
        try:
            state = dict(self.admission_controller() or {})
        except Exception:
            return {"blocked": False}
        state["blocked"] = bool(state.get("blocked", False))
        return state

    def _record_enqueue_locked(
        self,
        task: TextCpuTask,
        *,
        admission_wait_ms: float,
        backpressure_wait_ms: float,
        capacity_wait_ms: float,
    ) -> None:
        task.admission_wait_ms = float(max(0.0, admission_wait_ms))
        task.backpressure_wait_ms = float(max(0.0, backpressure_wait_ms))
        task.capacity_wait_ms = float(max(0.0, capacity_wait_ms))
        task.enqueued_at = time.perf_counter()
        task.pending_depth_on_enqueue = int(len(self.pending_tasks))
        self.pending_tasks.append(task)
        self.total_submitted += 1
        self.admission_wait_total_ms += task.admission_wait_ms
        self.admission_wait_peak_ms = max(self.admission_wait_peak_ms, task.admission_wait_ms)
        self.backpressure_wait_total_ms += task.backpressure_wait_ms
        self.backpressure_wait_peak_ms = max(self.backpressure_wait_peak_ms, task.backpressure_wait_ms)
        self.capacity_wait_total_ms += task.capacity_wait_ms
        self.capacity_wait_peak_ms = max(self.capacity_wait_peak_ms, task.capacity_wait_ms)
        if task.backpressure_wait_ms > 0.0:
            self.backpressure_blocked_total += 1
        if len(self.pending_tasks) > self.pending_peak:
            self.pending_peak = len(self.pending_tasks)
        self.condition.notify_all()

    async def _enqueue_task_async(self, task: TextCpuTask) -> None:
        admission_started = time.perf_counter()
        backpressure_wait_ms = 0.0
        capacity_wait_ms = 0.0
        while True:
            loop_start = time.perf_counter()
            admission_state = self._get_admission_state()
            blocked = bool(admission_state.get("blocked", False))
            with self.condition:
                if not blocked and self._can_enqueue_locked():
                    self._record_enqueue_locked(
                        task,
                        admission_wait_ms=(time.perf_counter() - admission_started) * 1000.0,
                        backpressure_wait_ms=backpressure_wait_ms,
                        capacity_wait_ms=capacity_wait_ms,
                    )
                    return
            await asyncio.sleep(self.admission_poll_s)
            waited_ms = (time.perf_counter() - loop_start) * 1000.0
            if blocked:
                backpressure_wait_ms += waited_ms
            else:
                capacity_wait_ms += waited_ms

    def submit(self, text: str, language: str) -> Tuple[Any, Dict[str, float]]:
        task = TextCpuTask(text=str(text), language=str(language))
        asyncio.run(self._enqueue_task_async(task))
        task.done_event.wait()
        if task.error is not None:
            raise task.error
        return task.result, dict(task.profile)

    async def submit_async(self, text: str, language: str) -> Tuple[Any, Dict[str, float]]:
        loop = asyncio.get_running_loop()
        task = TextCpuTask(
            text=str(text),
            language=str(language),
            done_loop=loop,
            done_future=loop.create_future(),
        )
        await self._enqueue_task_async(task)
        return await task.done_future

    @staticmethod
    def _resolve_done_future(task: TextCpuTask) -> None:
        if task.done_future is None or task.done_future.done():
            return
        if task.error is not None:
            task.done_future.set_exception(task.error)
            return
        task.done_future.set_result((task.result, dict(task.profile)))

    def _notify_task_done(self, task: TextCpuTask) -> None:
        task.done_event.set()
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_done_future, task)
        except RuntimeError:
            pass

    def snapshot(self) -> Dict[str, int | float]:
        with self.condition:
            return {
                "worker_count": int(self.worker_count),
                "pending": int(len(self.pending_tasks)),
                "pending_peak": int(self.pending_peak),
                "active_workers": int(self.active_workers),
                "active_workers_peak": int(self.active_workers_peak),
                "total_submitted": int(self.total_submitted),
                "total_finished": int(self.total_finished),
                "max_pending_tasks": int(self.max_pending_tasks),
                "admission_wait_total_ms": float(self.admission_wait_total_ms),
                "admission_wait_peak_ms": float(self.admission_wait_peak_ms),
                "backpressure_wait_total_ms": float(self.backpressure_wait_total_ms),
                "backpressure_wait_peak_ms": float(self.backpressure_wait_peak_ms),
                "capacity_wait_total_ms": float(self.capacity_wait_total_ms),
                "capacity_wait_peak_ms": float(self.capacity_wait_peak_ms),
                "backpressure_blocked_total": int(self.backpressure_blocked_total),
            }

    def _run_loop(self) -> None:
        while True:
            with self.condition:
                while not self.pending_tasks:
                    self.condition.wait()
                task = self.pending_tasks.popleft()
                self.active_workers += 1
                self.active_workers_peak = max(self.active_workers_peak, self.active_workers)
            started_at = time.perf_counter()
            try:
                task.result = self.process_fn(task.text, task.language)
                task.profile = {
                    "text_cpu_admission_wait_ms": float(task.admission_wait_ms),
                    "text_cpu_backpressure_wait_ms": float(task.backpressure_wait_ms),
                    "text_cpu_capacity_wait_ms": float(task.capacity_wait_ms),
                    "text_cpu_queue_wait_ms": max(0.0, (started_at - task.enqueued_at) * 1000.0),
                    "text_cpu_pending_depth_on_enqueue": float(task.pending_depth_on_enqueue),
                    "text_cpu_run_ms": max(0.0, (time.perf_counter() - started_at) * 1000.0),
                }
            except Exception as exc:  # noqa: PERF203
                task.error = exc
            finally:
                with self.condition:
                    self.active_workers = max(0, self.active_workers - 1)
                    self.total_finished += 1
                    self.condition.notify_all()
                self._notify_task_done(task)
