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
    done_event: threading.Event = field(default_factory=threading.Event)
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
    ):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.stage_limiter = stage_limiter
        self.batch_window_s = max(0.0, float(batch_window_ms) / 1000.0)
        self.max_batch_items = max(1, int(max_batch_items))
        self.max_batch_tokens = max(16, int(max_batch_tokens))

        self.condition = threading.Condition()
        self.pending_tasks: Deque[BertFeatureTask] = deque()
        self.pending_peak = 0
        self.total_submitted = 0
        self.total_finished = 0
        self.total_batches = 0
        self.active_batch_size = 0
        self.active_batch_peak = 0
        self.worker_thread = threading.Thread(target=self._run_loop, name="prepare-bert-batch-worker", daemon=True)
        self.worker_thread.start()

    def _estimate_task_tokens(self, task: BertFeatureTask) -> int:
        return max(1, len(task.norm_text) + 2)

    def submit(self, norm_text: str, word2ph: List[int]) -> Tuple[torch.Tensor, Dict[str, float]]:
        task = BertFeatureTask(norm_text=str(norm_text), word2ph=list(word2ph))
        with self.condition:
            self.pending_tasks.append(task)
            self.total_submitted += 1
            if len(self.pending_tasks) > self.pending_peak:
                self.pending_peak = len(self.pending_tasks)
            self.condition.notify_all()
        task.done_event.wait()
        if task.error is not None:
            raise task.error
        assert task.result_feature is not None
        return task.result_feature, dict(task.profile)

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
                "batch_window_ms": int(self.batch_window_s * 1000.0),
                "max_batch_items": self.max_batch_items,
                "max_batch_tokens": self.max_batch_tokens,
            }

    def _collect_batch(self) -> List[BertFeatureTask]:
        with self.condition:
            while not self.pending_tasks:
                self.condition.wait()

            batch: List[BertFeatureTask] = [self.pending_tasks.popleft()]
            batch_tokens = self._estimate_task_tokens(batch[0])
            deadline = time.perf_counter() + self.batch_window_s

            while len(batch) < self.max_batch_items:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                if not self.pending_tasks:
                    self.condition.wait(timeout=remaining)
                    continue
                next_task = self.pending_tasks[0]
                next_tokens = self._estimate_task_tokens(next_task)
                if len(batch) >= self.max_batch_items or (batch_tokens + next_tokens) > self.max_batch_tokens:
                    break
                batch.append(self.pending_tasks.popleft())
                batch_tokens += next_tokens

            self.active_batch_size = len(batch)
            if self.active_batch_size > self.active_batch_peak:
                self.active_batch_peak = self.active_batch_size
            return batch

    def _finalize_batch(self, batch: List[BertFeatureTask]) -> None:
        with self.condition:
            self.active_batch_size = 0
            self.total_batches += 1
            self.total_finished += len(batch)

    def _run_batch(self, batch: List[BertFeatureTask]) -> None:
        batch_started = time.perf_counter()
        texts = [task.norm_text for task in batch]
        batch_tokens = sum(self._estimate_task_tokens(task) for task in batch)

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
                    "bert_forward_ms": float(forward_ms),
                    "bert_tokenize_ms": float(tokenize_ms),
                    "bert_scatter_ms": 0.0,
                    "bert_calls": 1.0,
                    "bert_stage_slots": float(limiter_stats["slots"]),
                    "bert_stage_inflight_peak": float(limiter_stats["peak_inflight"]),
                    "bert_batch_size": float(len(batch)),
                    "bert_batch_tokens": float(batch_tokens),
                }
            except Exception as exc:  # noqa: PERF203
                task.error = exc
        scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
        for task in batch:
            if task.result_feature is not None:
                task.profile["bert_scatter_ms"] = float(scatter_ms)
            task.done_event.set()

    def _run_loop(self) -> None:
        while True:
            batch = self._collect_batch()
            try:
                self._run_batch(batch)
            except Exception as exc:  # noqa: PERF203
                for task in batch:
                    task.error = exc
                    task.done_event.set()
            finally:
                self._finalize_batch(batch)
