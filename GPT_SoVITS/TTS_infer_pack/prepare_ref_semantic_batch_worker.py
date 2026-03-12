import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import librosa
import numpy as np
import torch


REF_AUDIO_MIN_SAMPLES_16K = 48000
REF_AUDIO_MAX_SAMPLES_16K = 160000


def prepare_prompt_semantic_wav16k(raw_audio: torch.Tensor, raw_sr: int, zero_wav_samples: int) -> torch.Tensor:
    wav_mono = raw_audio
    if wav_mono.dim() == 2 and wav_mono.shape[0] != 1:
        wav_mono = wav_mono.mean(0, keepdim=True)
    wav16k = wav_mono.squeeze(0).cpu().numpy()
    if raw_sr != 16000:
        wav16k = librosa.resample(wav16k, orig_sr=raw_sr, target_sr=16000)
    if wav16k.shape[0] > REF_AUDIO_MAX_SAMPLES_16K or wav16k.shape[0] < REF_AUDIO_MIN_SAMPLES_16K:
        raise OSError("参考音频在3~10秒范围外，请更换！")
    wav16k = np.ascontiguousarray(wav16k, dtype=np.float32)
    if zero_wav_samples > 0:
        wav16k = np.concatenate([wav16k, np.zeros(int(zero_wav_samples), dtype=np.float32)], axis=0)
    return torch.from_numpy(wav16k)


def conv1d_output_lengths(input_lengths: torch.Tensor, conv1d: torch.nn.Conv1d | None) -> torch.Tensor:
    if conv1d is None:
        return input_lengths.to(dtype=torch.long)
    kernel_size = int(conv1d.kernel_size[0])
    stride = int(conv1d.stride[0])
    padding = int(conv1d.padding[0])
    dilation = int(conv1d.dilation[0])
    output_lengths = torch.div(
        input_lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride,
        rounding_mode="floor",
    ) + 1
    return torch.clamp(output_lengths, min=0).to(dtype=torch.long)


@dataclass
class RefSemanticTask:
    raw_audio: torch.Tensor
    raw_sr: int
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.perf_counter)
    batch_popped_at: float = 0.0
    done_event: threading.Event = field(default_factory=threading.Event)
    done_loop: asyncio.AbstractEventLoop | None = None
    done_future: asyncio.Future | None = None
    result_prompt_semantic: torch.Tensor | None = None
    error: Exception | None = None
    profile: Dict[str, float] = field(default_factory=dict)


class PrepareRefSemanticBatchWorker:
    def __init__(
        self,
        ssl_model,
        vits_model,
        device,
        is_half: bool,
        zero_wav_samples: int,
        stage_limiter=None,
        batch_window_ms: int = 5,
        max_batch_items: int = 8,
        max_batch_samples: int = 960000,
    ):
        self.ssl_model = ssl_model
        self.vits_model = vits_model
        self.device = device
        self.is_half = bool(is_half)
        self.zero_wav_samples = max(0, int(zero_wav_samples))
        self.stage_limiter = stage_limiter
        self.batch_window_s = max(0.0, float(batch_window_ms) / 1000.0)
        self.max_batch_items = max(1, int(max_batch_items))
        self.max_batch_samples = max(REF_AUDIO_MIN_SAMPLES_16K + self.zero_wav_samples, int(max_batch_samples))

        self.condition = threading.Condition()
        self.pending_tasks: Deque[RefSemanticTask] = deque()
        self.pending_peak = 0
        self.total_submitted = 0
        self.total_finished = 0
        self.total_batches = 0
        self.active_batch_size = 0
        self.active_batch_peak = 0
        self.active_batch_samples = 0
        self.active_batch_samples_peak = 0
        self.worker_thread = threading.Thread(
            target=self._run_loop,
            name="prepare-ref-semantic-batch-worker",
            daemon=True,
        )
        self.worker_thread.start()

    def _estimate_task_samples(self, task: RefSemanticTask) -> int:
        raw_len = int(task.raw_audio.shape[-1]) if task.raw_audio.dim() > 0 else 0
        base = int(round(raw_len * 16000.0 / max(1, int(task.raw_sr))))
        return max(REF_AUDIO_MIN_SAMPLES_16K, base) + self.zero_wav_samples

    def submit(self, raw_audio: torch.Tensor, raw_sr: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        task = RefSemanticTask(raw_audio=raw_audio, raw_sr=int(raw_sr))
        with self.condition:
            self.pending_tasks.append(task)
            self.total_submitted += 1
            if len(self.pending_tasks) > self.pending_peak:
                self.pending_peak = len(self.pending_tasks)
            self.condition.notify_all()
        task.done_event.wait()
        if task.error is not None:
            raise task.error
        assert task.result_prompt_semantic is not None
        return task.result_prompt_semantic, dict(task.profile)

    async def submit_async(self, raw_audio: torch.Tensor, raw_sr: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        loop = asyncio.get_running_loop()
        task = RefSemanticTask(
            raw_audio=raw_audio,
            raw_sr=int(raw_sr),
            done_loop=loop,
            done_future=loop.create_future(),
        )
        with self.condition:
            self.pending_tasks.append(task)
            self.total_submitted += 1
            if len(self.pending_tasks) > self.pending_peak:
                self.pending_peak = len(self.pending_tasks)
            self.condition.notify_all()
        return await task.done_future

    @staticmethod
    def _resolve_done_future(task: RefSemanticTask) -> None:
        if task.done_future is None or task.done_future.done():
            return
        if task.error is not None:
            task.done_future.set_exception(task.error)
            return
        assert task.result_prompt_semantic is not None
        task.done_future.set_result((task.result_prompt_semantic, dict(task.profile)))

    def _notify_task_done(self, task: RefSemanticTask) -> None:
        task.done_event.set()
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_done_future, task)
        except RuntimeError:
            pass

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
                "active_batch_samples": self.active_batch_samples,
                "active_batch_samples_peak": self.active_batch_samples_peak,
                "batch_window_ms": int(self.batch_window_s * 1000.0),
                "max_batch_items": self.max_batch_items,
                "max_batch_samples": self.max_batch_samples,
            }

    def _collect_batch(self) -> tuple[List[RefSemanticTask], float]:
        with self.condition:
            while not self.pending_tasks:
                self.condition.wait()

            first_task = self.pending_tasks.popleft()
            first_task.batch_popped_at = time.perf_counter()
            batch: List[RefSemanticTask] = [first_task]
            batch_samples = self._estimate_task_samples(batch[0])
            deadline = time.perf_counter() + self.batch_window_s

            while len(batch) < self.max_batch_items:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                if not self.pending_tasks:
                    self.condition.wait(timeout=remaining)
                    continue
                next_task = self.pending_tasks[0]
                next_samples = self._estimate_task_samples(next_task)
                if len(batch) >= self.max_batch_items or (batch_samples + next_samples) > self.max_batch_samples:
                    break
                popped_task = self.pending_tasks.popleft()
                popped_task.batch_popped_at = time.perf_counter()
                batch.append(popped_task)
                batch_samples += next_samples

            self.active_batch_size = len(batch)
            self.active_batch_samples = batch_samples
            if self.active_batch_size > self.active_batch_peak:
                self.active_batch_peak = self.active_batch_size
            if self.active_batch_samples > self.active_batch_samples_peak:
                self.active_batch_samples_peak = self.active_batch_samples
            return batch, time.perf_counter()

    def _finalize_batch(self, batch: List[RefSemanticTask]) -> None:
        with self.condition:
            self.active_batch_size = 0
            self.active_batch_samples = 0
            self.total_batches += 1
            self.total_finished += len(batch)

    def _get_hidden_lengths(self, attention_mask: torch.Tensor, hidden_length: int) -> torch.Tensor:
        model = self.ssl_model.model
        if hasattr(model, "_get_feature_vector_attention_mask"):
            feature_mask = model._get_feature_vector_attention_mask(hidden_length, attention_mask)
            return feature_mask.to(dtype=torch.long).sum(dim=1)
        raw_lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
        if hasattr(model, "_get_feat_extract_output_lengths"):
            return model._get_feat_extract_output_lengths(raw_lengths).to(dtype=torch.long)
        return torch.full((attention_mask.shape[0],), int(hidden_length), dtype=torch.long, device=attention_mask.device)

    @torch.inference_mode()
    def _run_batch(self, batch: List[RefSemanticTask], batch_collected_at: float) -> None:
        batch_started = time.perf_counter()
        prepared_start = time.perf_counter()
        prepared_wavs = [
            prepare_prompt_semantic_wav16k(task.raw_audio, int(task.raw_sr), self.zero_wav_samples) for task in batch
        ]
        cpu_prepare_ms = (time.perf_counter() - prepared_start) * 1000.0
        wav_lengths = torch.tensor([int(wav.shape[0]) for wav in prepared_wavs], dtype=torch.long)
        batch_samples = int(wav_lengths.sum().item())
        max_wav_len = int(wav_lengths.max().item())

        input_values_cpu = torch.zeros((len(batch), max_wav_len), dtype=torch.float32)
        attention_mask_cpu = torch.zeros((len(batch), max_wav_len), dtype=torch.long)
        for batch_index, wav in enumerate(prepared_wavs):
            wav_len = int(wav.shape[0])
            input_values_cpu[batch_index, :wav_len] = wav
            attention_mask_cpu[batch_index, :wav_len] = 1

        limiter_stats = {"wait_ms": 0.0, "peak_inflight": 1, "slots": 0}
        if self.stage_limiter is None:
            input_values = input_values_cpu.to(self.device)
            attention_mask = attention_mask_cpu.to(self.device)
            if self.is_half:
                input_values = input_values.half()
            forward_start = time.perf_counter()
            outputs = self.ssl_model.model(input_values, attention_mask=attention_mask)
            hubert_feature = outputs["last_hidden_state"].transpose(1, 2)
            hidden_lengths = self._get_hidden_lengths(attention_mask, int(hubert_feature.shape[-1]))
            codes = self.vits_model.extract_latent(hubert_feature)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
        else:
            with self.stage_limiter.enter() as limiter_stats:
                input_values = input_values_cpu.to(self.device)
                attention_mask = attention_mask_cpu.to(self.device)
                if self.is_half:
                    input_values = input_values.half()
                forward_start = time.perf_counter()
                outputs = self.ssl_model.model(input_values, attention_mask=attention_mask)
                hubert_feature = outputs["last_hidden_state"].transpose(1, 2)
                hidden_lengths = self._get_hidden_lengths(attention_mask, int(hubert_feature.shape[-1]))
                codes = self.vits_model.extract_latent(hubert_feature)
                forward_ms = (time.perf_counter() - forward_start) * 1000.0

        code_lengths = conv1d_output_lengths(hidden_lengths.detach().cpu(), getattr(self.vits_model, "ssl_proj", None))
        scatter_start = time.perf_counter()
        for batch_index, task in enumerate(batch):
            try:
                code_len = int(code_lengths[batch_index].item())
                task.result_prompt_semantic = codes[batch_index, 0, :code_len].detach().clone()
                worker_queue_wait_ms = max(0.0, (float(task.batch_popped_at) - float(task.created_at)) * 1000.0)
                batch_collect_wait_ms = max(0.0, (float(batch_collected_at) - float(task.batch_popped_at)) * 1000.0)
                stage_limiter_wait_ms = float(limiter_stats["wait_ms"])
                task.profile = {
                    "prompt_semantic_wait_ms": worker_queue_wait_ms
                    + batch_collect_wait_ms
                    + stage_limiter_wait_ms,
                    "prompt_semantic_worker_queue_wait_ms": worker_queue_wait_ms,
                    "prompt_semantic_batch_collect_wait_ms": batch_collect_wait_ms,
                    "prompt_semantic_stage_limiter_wait_ms": stage_limiter_wait_ms,
                    "prompt_semantic_batch_dispatch_delay_ms": max(
                        0.0, (float(batch_started) - float(batch_collected_at)) * 1000.0
                    ),
                    "prompt_semantic_cpu_prepare_ms": float(cpu_prepare_ms),
                    "prompt_semantic_forward_ms": float(forward_ms),
                    "prompt_semantic_scatter_ms": 0.0,
                    "prompt_semantic_calls": 1.0,
                    "prompt_semantic_stage_slots": float(limiter_stats["slots"]),
                    "prompt_semantic_stage_inflight_peak": float(limiter_stats["peak_inflight"]),
                    "prompt_semantic_batch_size": float(len(batch)),
                    "prompt_semantic_batch_samples": float(batch_samples),
                }
            except Exception as exc:  # noqa: PERF203
                task.error = exc
        scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
        for task in batch:
            if task.result_prompt_semantic is not None:
                task.profile["prompt_semantic_scatter_ms"] = float(scatter_ms)
            self._notify_task_done(task)

    def _run_loop(self) -> None:
        while True:
            batch, batch_collected_at = self._collect_batch()
            try:
                self._run_batch(batch, batch_collected_at)
            except Exception as exc:  # noqa: PERF203
                for task in batch:
                    task.error = exc
                    self._notify_task_done(task)
            finally:
                self._finalize_batch(batch)
