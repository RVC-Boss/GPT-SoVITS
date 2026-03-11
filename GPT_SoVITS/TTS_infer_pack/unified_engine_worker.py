from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np
import torch

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import PrepareCoordinator, PreparedCpuStage
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SActiveBatch, T2SFinishedItem, T2SRequestState, decode_one_step, merge_active_batches, run_prefill_active_batch, run_scheduler_continuous
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineStatus, RuntimeStateCallbacks, SchedulerFinalizeTask, SchedulerJobRegistry, SchedulerPendingJob


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


