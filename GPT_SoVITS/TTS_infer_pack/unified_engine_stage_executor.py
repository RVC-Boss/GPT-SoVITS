from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SFinishedItem, T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import (
    EngineDecodeRuntimeOwner,
    EngineDispatchTask,
    EngineGpuPrepareTask,
    EngineStatus,
    EngineTaskQueueOwner,
    SchedulerFinalizeTask,
    SchedulerPendingJob,
)
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker import UnifiedSchedulerWorker


class EngineStageExecutor:
    def __init__(
        self,
        *,
        tts: TTS,
        scheduler_worker: UnifiedSchedulerWorker,
        prepare_queue_owner: EngineTaskQueueOwner,
        finalize_queue_owner: EngineTaskQueueOwner,
        dispatch_queue_owner: EngineTaskQueueOwner,
        decode_runtime_owner: EngineDecodeRuntimeOwner,
        update_request_state: Callable[[str, str, Optional[Dict[str, Any]]], None],
        merge_request_state_profile: Callable[[str, Optional[Dict[str, Any]]], None],
        fail_request_state: Callable[[str, str], None],
        get_engine_job: Callable[[str], SchedulerPendingJob | None],
        register_engine_job: Callable[[SchedulerPendingJob], None],
        fail_engine_jobs: Callable[[List[str], str], None],
        complete_engine_job: Callable[..., None],
        add_engine_prefill_time: Callable[[List[SchedulerPendingJob], float], None],
        add_engine_merge_time: Callable[[List[str], float], None],
        add_engine_decode_time: Callable[[List[str], float], None],
        enqueue_engine_finished_items: Callable[[List[T2SFinishedItem]], None],
        snapshot_engine_dispatch_state: Callable[[], Dict[str, Any]],
        snapshot_engine_decode_runtime_state: Callable[[], Dict[str, Any]],
    ) -> None:
        self.tts = tts
        self.scheduler_worker = scheduler_worker
        self.prepare_queue_owner = prepare_queue_owner
        self.finalize_queue_owner = finalize_queue_owner
        self.dispatch_queue_owner = dispatch_queue_owner
        self.decode_runtime_owner = decode_runtime_owner
        self.update_request_state = update_request_state
        self.merge_request_state_profile = merge_request_state_profile
        self.fail_request_state = fail_request_state
        self.get_engine_job = get_engine_job
        self.register_engine_job = register_engine_job
        self.fail_engine_jobs = fail_engine_jobs
        self.complete_engine_job = complete_engine_job
        self.add_engine_prefill_time = add_engine_prefill_time
        self.add_engine_merge_time = add_engine_merge_time
        self.add_engine_decode_time = add_engine_decode_time
        self.enqueue_engine_finished_items = enqueue_engine_finished_items
        self.snapshot_engine_dispatch_state = snapshot_engine_dispatch_state
        self.snapshot_engine_decode_runtime_state = snapshot_engine_decode_runtime_state
        self._notify_arbiter: Callable[[], None] | None = None

    def bind_notify_arbiter(self, notify_arbiter: Callable[[], None]) -> None:
        self._notify_arbiter = notify_arbiter

    def notify_arbiter(self) -> None:
        if self._notify_arbiter is not None:
            self._notify_arbiter()

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

    async def prepare_state_via_engine_gpu_queue(
        self,
        *,
        spec: Any,
        prepare_submit_at: float,
        engine_request_id: str | None,
    ) -> tuple[T2SRequestState, float, float]:
        cpu_stage = await self.scheduler_worker.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)
        if engine_request_id not in [None, ""]:
            self.update_request_state(
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
        self.prepare_queue_owner.enqueue(task)
        self.notify_arbiter()
        return await done_future

    def enqueue_worker_finished_for_finalize(self, tasks: List[SchedulerFinalizeTask]) -> None:
        if not tasks:
            return
        for task in tasks:
            job = self.get_engine_job(task.request_id)
            if job is not None:
                self.update_request_state(
                    job.engine_request_id,
                    EngineStatus.READY_FOR_FINALIZE,
                    {
                        "finish_reason": task.item.finish_reason,
                        "semantic_len": int(task.item.semantic_tokens.shape[0]),
                        "finish_idx": int(task.item.finish_idx),
                    },
                )
        self.finalize_queue_owner.enqueue_many(tasks)
        self.notify_arbiter()

    def take_engine_finalize_batch_nonblocking(self) -> List[SchedulerFinalizeTask]:
        finalize_policy = self.scheduler_worker.get_finalize_batch_policy()
        return self.finalize_queue_owner.take_finalize_batch(
            finalize_mode=str(finalize_policy.get("finalize_mode", "async")),
            batch_max_items=int(finalize_policy.get("finalize_batch_max_items", 1)),
            batch_wait_s=float(finalize_policy.get("finalize_batch_wait_s", 0.0)),
            use_vocoder=bool(self.tts.configs.use_vocoder),
        )

    async def enqueue_prepared_state_for_dispatch(
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
        self.dispatch_queue_owner.enqueue(task)
        self.notify_arbiter()
        self.merge_request_state_profile(
            task.engine_request_id or task.request_id,
            {
                "engine_dispatch_queue_depth_on_enqueue": int(
                    self.snapshot_engine_dispatch_state()["waiting_count"]
                ),
            },
        )
        return task

    def run_engine_prepare_once(self) -> bool:
        task = self.prepare_queue_owner.pop_left()
        if task is None:
            return False
        queue_wait_ms = max(0.0, (time.perf_counter() - task.enqueue_time) * 1000.0)
        try:
            state, prepare_exec_started_at, prepare_exec_finished_at = asyncio.run(
                self.scheduler_worker.prepare_gpu_stage_profiled_async(task.cpu_stage)
            )
            state.prepare_profile["engine_gpu_prepare_queue_wait_ms"] = float(queue_wait_ms)
            if task.engine_request_id not in [None, ""]:
                self.merge_request_state_profile(
                    str(task.engine_request_id),
                    {"engine_gpu_prepare_queue_wait_ms": float(queue_wait_ms)},
                )
            self.prepare_queue_owner.mark_completed(1)
            self._notify_prepare_result(task, (state, prepare_exec_started_at, prepare_exec_finished_at))
            return True
        except Exception as exc:
            task.error = str(exc)
            self.fail_request_state(task.engine_request_id or task.request_id, str(exc))
            self._notify_prepare_error(task, exc)
            return True

    def run_engine_finalize_once(self) -> bool:
        tasks = self.take_engine_finalize_batch_nonblocking()
        if not tasks:
            return False
        self.scheduler_worker.begin_finalize_execution(len(tasks))
        try:
            jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]] = []
            for task in tasks:
                job = self.get_engine_job(task.request_id)
                if job is None:
                    continue
                jobs_and_items.append((job, task.item))
            if not jobs_and_items:
                return False
            now = time.perf_counter()
            for task in tasks:
                job = self.get_engine_job(task.request_id)
                if job is not None:
                    job.finalize_wait_ms += max(0.0, (now - task.enqueued_time) * 1000.0)
            for job, item in jobs_and_items:
                self.update_request_state(
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
                self.complete_engine_job(job, item, sample_rate=sample_rate, audio_data=audio_data)
        except Exception as exc:
            self.fail_engine_jobs([task.request_id for task in tasks], str(exc))
        finally:
            self.scheduler_worker.end_finalize_execution(len(tasks))
        self.finalize_queue_owner.mark_completed(len(tasks), notify=True)
        return True

    def run_engine_dispatch_once(self, policy_snapshot: Dict[str, Any], worker_state: Dict[str, Any]) -> bool:
        if not bool(policy_snapshot.get("allowed", True)):
            return False
        dispatch_task = self.dispatch_queue_owner.pop_left()
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
            self.register_engine_job(worker_job)
            if self.scheduler_worker.is_engine_decode_control_enabled():
                self.decode_runtime_owner.enqueue_pending_job(worker_job)
                self.notify_arbiter()
            self.dispatch_queue_owner.mark_completed(1)
            return True
        except Exception as exc:
            dispatch_task.error = str(exc)
            self.fail_request_state(dispatch_task.engine_request_id or dispatch_task.request_id, str(exc))
            self._notify_dispatch_error(dispatch_task, exc)
            return True

    def run_engine_decode_runtime_once(self) -> bool:
        if not self.scheduler_worker.is_engine_decode_control_enabled():
            return False
        runtime_state = self.snapshot_engine_decode_runtime_state()
        pending_jobs = self.decode_runtime_owner.take_pending_jobs_nonblocking(
            wait_for_batch=int(runtime_state.get("active_request_count", 0)) <= 0
        )
        result = self.scheduler_worker.execute_decode_cycle(
            pending_jobs=pending_jobs,
            active_batch=self.decode_runtime_owner.get_active_batch(),
            external_bookkeeping=True,
        )
        prefill_phase = dict(result.get("prefill_phase") or {})
        if prefill_phase.get("error"):
            self.fail_engine_jobs(list(prefill_phase.get("error_request_ids") or []), str(prefill_phase.get("error")))
        else:
            prefill_jobs = list(prefill_phase.get("pending_jobs") or [])
            self.add_engine_prefill_time(prefill_jobs, float(prefill_phase.get("prefill_elapsed_s", 0.0)))
            self.add_engine_merge_time(
                [] if result.get("active_batch") is None else list(result["active_batch"].request_ids),
                float(prefill_phase.get("merge_elapsed_s", 0.0)),
            )
            self.enqueue_engine_finished_items(list(prefill_phase.get("finished_items") or []))
        decode_phase = dict(result.get("decode_phase") or {})
        if decode_phase.get("error"):
            self.fail_engine_jobs(list(decode_phase.get("error_request_ids") or []), str(decode_phase.get("error")))
        else:
            self.add_engine_decode_time(
                list(decode_phase.get("request_ids") or []),
                float(decode_phase.get("decode_elapsed_s", 0.0)),
            )
            self.enqueue_engine_finished_items(list(decode_phase.get("finished_items") or []))
        self.decode_runtime_owner.set_active_batch(result.get("active_batch"))
        if result.get("executed", False):
            self.decode_runtime_owner.refresh_state("engine_decode_cycle")
        return bool(result.get("executed", False))
