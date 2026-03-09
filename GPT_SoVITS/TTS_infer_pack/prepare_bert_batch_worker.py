import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import torch


@dataclass
class BertFeatureTask:
    norm_text: str
    word2ph: List[int]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.perf_counter)
    enqueued_at: float = 0.0
    admission_wait_ms: float = 0.0
    pending_depth_on_enqueue: int = 0
    done_event: threading.Event = field(default_factory=threading.Event)
    done_loop: asyncio.AbstractEventLoop | None = None
    done_future: asyncio.Future | None = None
    result_feature: torch.Tensor | None = None
    error: Exception | None = None
    profile: Dict[str, float] = field(default_factory=dict)


class PrepareBertBatchWorker:
    def __init__(
        self,
        bert_model,
        tokenizer,
        device,
        stage_limiter=None,
        batch_window_ms: int = 5,
        max_batch_items: int = 16,
        max_batch_tokens: int = 4096,
        max_pending_tasks: int = 0,
        admission_poll_ms: int = 1,
        high_pressure_pending_threshold: int = 0,
        high_pressure_batch_window_ms: int | None = None,
        high_pressure_max_batch_items: int | None = None,
        high_pressure_max_batch_tokens: int | None = None,
    ):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.stage_limiter = stage_limiter
        self.batch_window_ms = max(0, int(batch_window_ms))
        self.batch_window_s = float(self.batch_window_ms) / 1000.0
        self.max_batch_items = max(1, int(max_batch_items))
        self.max_batch_tokens = max(16, int(max_batch_tokens))
        self.max_pending_tasks = max(0, int(max_pending_tasks))
        self.admission_poll_s = max(0.0005, float(max(1, int(admission_poll_ms))) / 1000.0)

        self.high_pressure_pending_threshold = max(
            0,
            int(high_pressure_pending_threshold)
            if int(high_pressure_pending_threshold) > 0
            else max(self.max_batch_items * 2, 32),
        )
        hp_window_ms = self.batch_window_ms if high_pressure_batch_window_ms is None else int(high_pressure_batch_window_ms)
        hp_items = self.max_batch_items if high_pressure_max_batch_items is None else int(high_pressure_max_batch_items)
        hp_tokens = self.max_batch_tokens if high_pressure_max_batch_tokens is None else int(high_pressure_max_batch_tokens)
        self.high_pressure_batch_window_ms = max(0, hp_window_ms)
        self.high_pressure_batch_window_s = float(self.high_pressure_batch_window_ms) / 1000.0
        self.high_pressure_max_batch_items = max(self.max_batch_items, hp_items)
        self.high_pressure_max_batch_tokens = max(self.max_batch_tokens, hp_tokens)

        self.condition = threading.Condition()
        self.pending_tasks: Deque[BertFeatureTask] = deque()
        self.pending_peak = 0
        self.total_submitted = 0
        self.total_finished = 0
        self.total_batches = 0
        self.active_batch_size = 0
        self.active_batch_peak = 0
        self.active_batch_tokens = 0
        self.active_batch_tokens_peak = 0
        self.high_pressure_batches = 0
        self.admission_wait_total_ms = 0.0
        self.admission_wait_peak_ms = 0.0
        self.worker_thread = threading.Thread(target=self._run_loop, name="prepare-bert-batch-worker", daemon=True)
        self.worker_thread.start()

    def _estimate_task_tokens(self, task: BertFeatureTask) -> int:
        return max(1, len(task.norm_text) + 2)

    def _can_enqueue_locked(self) -> bool:
        if self.max_pending_tasks <= 0:
            return True
        return (len(self.pending_tasks) + self.active_batch_size) < self.max_pending_tasks

    def _record_enqueue_locked(self, task: BertFeatureTask, admission_wait_ms: float) -> None:
        task.admission_wait_ms = float(max(0.0, admission_wait_ms))
        task.enqueued_at = time.perf_counter()
        task.pending_depth_on_enqueue = int(len(self.pending_tasks))
        self.pending_tasks.append(task)
        self.total_submitted += 1
        self.admission_wait_total_ms += task.admission_wait_ms
        self.admission_wait_peak_ms = max(self.admission_wait_peak_ms, task.admission_wait_ms)
        if len(self.pending_tasks) > self.pending_peak:
            self.pending_peak = len(self.pending_tasks)
        self.condition.notify_all()

    def _enqueue_task(self, task: BertFeatureTask) -> None:
        admission_started = time.perf_counter()
        with self.condition:
            while not self._can_enqueue_locked():
                self.condition.wait(timeout=self.admission_poll_s)
            self._record_enqueue_locked(task, (time.perf_counter() - admission_started) * 1000.0)

    async def _enqueue_task_async(self, task: BertFeatureTask) -> None:
        admission_started = time.perf_counter()
        while True:
            with self.condition:
                if self._can_enqueue_locked():
                    self._record_enqueue_locked(task, (time.perf_counter() - admission_started) * 1000.0)
                    return
            await asyncio.sleep(self.admission_poll_s)

    def submit(self, norm_text: str, word2ph: List[int]) -> Tuple[torch.Tensor, Dict[str, float]]:
        task = BertFeatureTask(norm_text=str(norm_text), word2ph=list(word2ph))
        self._enqueue_task(task)
        task.done_event.wait()
        if task.error is not None:
            raise task.error
        assert task.result_feature is not None
        return task.result_feature, dict(task.profile)

    async def submit_async(self, norm_text: str, word2ph: List[int]) -> Tuple[torch.Tensor, Dict[str, float]]:
        loop = asyncio.get_running_loop()
        task = BertFeatureTask(
            norm_text=str(norm_text),
            word2ph=list(word2ph),
            done_loop=loop,
            done_future=loop.create_future(),
        )
        await self._enqueue_task_async(task)
        return await task.done_future

    def snapshot(self) -> Dict[str, int]:
        with self.condition:
            return {
                "pending": len(self.pending_tasks),
                "pending_peak": self.pending_peak,
                "total_submitted": self.total_submitted,
                "total_finished": self.total_finished,
                "total_batches": self.total_batches,
                "active_batch_size": self.active_batch_size,
                "active_batch_peak": self.active_batch_peak,
                "active_batch_tokens": self.active_batch_tokens,
                "active_batch_tokens_peak": self.active_batch_tokens_peak,
                "batch_window_ms": int(self.batch_window_s * 1000.0),
                "max_batch_items": self.max_batch_items,
                "max_batch_tokens": self.max_batch_tokens,
                "max_pending_tasks": self.max_pending_tasks,
                "high_pressure_pending_threshold": self.high_pressure_pending_threshold,
                "high_pressure_batch_window_ms": self.high_pressure_batch_window_ms,
                "high_pressure_max_batch_items": self.high_pressure_max_batch_items,
                "high_pressure_max_batch_tokens": self.high_pressure_max_batch_tokens,
                "high_pressure_batches": self.high_pressure_batches,
                "admission_wait_total_ms": self.admission_wait_total_ms,
                "admission_wait_peak_ms": self.admission_wait_peak_ms,
            }

    def _select_batch_policy_locked(self) -> Tuple[float, int, int, bool, int]:
        pending_depth = len(self.pending_tasks)
        use_high_pressure = (
            self.high_pressure_pending_threshold > 0
            and pending_depth >= self.high_pressure_pending_threshold
        )
        if use_high_pressure:
            return (
                self.high_pressure_batch_window_s,
                self.high_pressure_max_batch_items,
                self.high_pressure_max_batch_tokens,
                True,
                pending_depth,
            )
        return (
            self.batch_window_s,
            self.max_batch_items,
            self.max_batch_tokens,
            False,
            pending_depth,
        )

    def _collect_batch(self) -> Tuple[List[BertFeatureTask], Dict[str, float]]:
        with self.condition:
            while not self.pending_tasks:
                self.condition.wait()

            collect_started = time.perf_counter()
            batch_window_s, max_batch_items, max_batch_tokens, use_high_pressure, pending_depth_on_collect = (
                self._select_batch_policy_locked()
            )
            batch: List[BertFeatureTask] = [self.pending_tasks.popleft()]
            batch_tokens = self._estimate_task_tokens(batch[0])
            deadline = time.perf_counter() + batch_window_s

            while len(batch) < max_batch_items:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                if not self.pending_tasks:
                    self.condition.wait(timeout=remaining)
                    continue
                next_task = self.pending_tasks[0]
                next_tokens = self._estimate_task_tokens(next_task)
                if len(batch) >= max_batch_items or (batch_tokens + next_tokens) > max_batch_tokens:
                    break
                batch.append(self.pending_tasks.popleft())
                batch_tokens += next_tokens

            self.active_batch_size = len(batch)
            self.active_batch_tokens = batch_tokens
            if self.active_batch_size > self.active_batch_peak:
                self.active_batch_peak = self.active_batch_size
            if self.active_batch_tokens > self.active_batch_tokens_peak:
                self.active_batch_tokens_peak = self.active_batch_tokens
            if use_high_pressure:
                self.high_pressure_batches += 1
            return batch, {
                "collect_wait_ms": (time.perf_counter() - collect_started) * 1000.0,
                "batch_tokens": float(batch_tokens),
                "pending_depth_on_collect": float(pending_depth_on_collect),
                "high_pressure_mode": 1.0 if use_high_pressure else 0.0,
                "batch_window_ms": float(self.high_pressure_batch_window_ms if use_high_pressure else self.batch_window_ms),
            }

    def _finalize_batch(self, batch: List[BertFeatureTask]) -> None:
        with self.condition:
            self.active_batch_size = 0
            self.active_batch_tokens = 0
            self.total_batches += 1
            self.total_finished += len(batch)
            self.condition.notify_all()

    def _run_batch(self, batch: List[BertFeatureTask], batch_meta: Dict[str, float]) -> None:
        batch_started = time.perf_counter()
        texts = [task.norm_text for task in batch]
        batch_tokens = int(batch_meta["batch_tokens"])

        limiter_stats = {"wait_ms": 0.0, "peak_inflight": 1, "slots": 0}
        if self.stage_limiter is None:
            tokenize_start = time.perf_counter()
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
            tokenize_ms = (time.perf_counter() - tokenize_start) * 1000.0
            attention_mask_cpu = inputs["attention_mask"].cpu()
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            forward_start = time.perf_counter()
            with torch.no_grad():
                outputs = self.bert_model(**inputs, output_hidden_states=True)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
        else:
            with self.stage_limiter.enter() as limiter_stats:
                tokenize_start = time.perf_counter()
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
                tokenize_ms = (time.perf_counter() - tokenize_start) * 1000.0
                attention_mask_cpu = inputs["attention_mask"].cpu()
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                forward_start = time.perf_counter()
                with torch.no_grad():
                    outputs = self.bert_model(**inputs, output_hidden_states=True)
                forward_ms = (time.perf_counter() - forward_start) * 1000.0

        hidden = outputs["hidden_states"][-3].detach().cpu()
        scatter_start = time.perf_counter()
        for batch_index, task in enumerate(batch):
            try:
                text_len = len(task.word2ph)
                if text_len != len(task.norm_text):
                    raise AssertionError(
                        f"word2ph/text length mismatch: task={task.task_id} word2ph={text_len} text={len(task.norm_text)}"
                    )
                seq_len = int(attention_mask_cpu[batch_index].sum().item())
                char_features = hidden[batch_index, 1 : seq_len - 1]
                if char_features.shape[0] != text_len:
                    raise AssertionError(
                        f"bert token length mismatch: task={task.task_id} token_len={char_features.shape[0]} text_len={text_len}"
                    )
                phone_level_feature = []
                for char_index, repeat_count in enumerate(task.word2ph):
                    phone_level_feature.append(char_features[char_index].repeat(repeat_count, 1))
                task.result_feature = torch.cat(phone_level_feature, dim=0).T
                task.profile = {
                    "bert_wait_ms": (batch_started - task.created_at) * 1000.0 + float(limiter_stats["wait_ms"]),
                    "bert_admission_wait_ms": float(task.admission_wait_ms),
                    "bert_queue_wait_ms": max(0.0, (batch_started - task.enqueued_at) * 1000.0),
                    "bert_batch_collect_wait_ms": float(batch_meta["collect_wait_ms"]),
                    "bert_forward_ms": float(forward_ms),
                    "bert_tokenize_ms": float(tokenize_ms),
                    "bert_scatter_ms": 0.0,
                    "bert_calls": 1.0,
                    "bert_stage_slots": float(limiter_stats["slots"]),
                    "bert_stage_inflight_peak": float(limiter_stats["peak_inflight"]),
                    "bert_batch_size": float(len(batch)),
                    "bert_batch_tokens": float(batch_tokens),
                    "bert_pending_depth_on_enqueue": float(task.pending_depth_on_enqueue),
                    "bert_pending_depth_on_collect": float(batch_meta["pending_depth_on_collect"]),
                    "bert_high_pressure_mode": float(batch_meta["high_pressure_mode"]),
                    "bert_batch_window_ms": float(batch_meta["batch_window_ms"]),
                }
            except Exception as exc:  # noqa: PERF203
                task.error = exc
        scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
        for task in batch:
            if task.result_feature is not None:
                task.profile["bert_scatter_ms"] = float(scatter_ms)
            task.done_event.set()
            self._notify_done_future(task)

    @staticmethod
    def _resolve_done_future(task: BertFeatureTask) -> None:
        if task.done_future is None or task.done_future.done():
            return
        if task.error is not None:
            task.done_future.set_exception(task.error)
            return
        assert task.result_feature is not None
        task.done_future.set_result((task.result_feature, dict(task.profile)))

    def _notify_done_future(self, task: BertFeatureTask) -> None:
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_done_future, task)
        except RuntimeError:
            pass

    def _run_loop(self) -> None:
        while True:
            batch, batch_meta = self._collect_batch()
            try:
                self._run_batch(batch, batch_meta)
            except Exception as exc:  # noqa: PERF203
                for task in batch:
                    task.error = exc
                    task.done_event.set()
                    self._notify_done_future(task)
            finally:
                self._finalize_batch(batch)
