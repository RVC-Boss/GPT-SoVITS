from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineGpuPrepareTask, EngineStatus


class EnginePrepareStageMixin:
    async def _wait_prepare_queue_admission(self) -> float:
        soft_max = max(0, int(os.environ.get("GPTSOVITS_ENGINE_PREPARE_QUEUE_SOFT_MAX", "0")))
        if soft_max <= 0:
            return 0.0
        poll_s = max(
            0.0005,
            float(max(1, int(os.environ.get("GPTSOVITS_ENGINE_PREPARE_QUEUE_ADMISSION_POLL_MS", "1")))) / 1000.0,
        )
        wait_start = time.perf_counter()
        while self.prepare_queue_owner.waiting_count() >= soft_max:
            await asyncio.sleep(poll_s)
        return max(0.0, (time.perf_counter() - wait_start) * 1000.0)

    async def prepare_state_via_engine_gpu_queue(
        self,
        *,
        spec: Any,
        prepare_submit_at: float,
        engine_request_id: str | None,
    ) -> tuple[T2SRequestState, float, float]:
        prepare_queue_admission_wait_ms = await self._wait_prepare_queue_admission()
        cpu_stage = await self.scheduler_worker.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)
        if engine_request_id not in [None, ""]:
            self.update_request_state(
                str(engine_request_id),
                EngineStatus.GPU_PREPARING,
                {
                    "engine_prepare_queue_admission_wait_ms": float(prepare_queue_admission_wait_ms),
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
            admission_wait_ms=float(prepare_queue_admission_wait_ms),
        )
        self.prepare_queue_owner.enqueue(task)
        self.notify_arbiter()
        return await done_future

    def run_engine_prepare_once(self) -> bool:
        prepare_batch_policy = self.scheduler_worker.get_prepare_batch_policy()
        tasks = self.prepare_queue_owner.pop_left_many(int(prepare_batch_policy.get("prepare_batch_max_items", 1)))
        if not tasks:
            return False
        now = time.perf_counter()
        queue_wait_ms_list = [max(0.0, (now - task.enqueue_time) * 1000.0) for task in tasks]
        batch_results = asyncio.run(
            self.scheduler_worker.prepare_gpu_stages_profiled_async([task.cpu_stage for task in tasks])
        )
        completed_count = 0
        for task, queue_wait_ms, result in zip(tasks, queue_wait_ms_list, batch_results):
            if isinstance(result, Exception):
                task.error = str(result)
                self.fail_request_state(task.engine_request_id or task.request_id, str(result))
                self._notify_prepare_error(task, result)
                completed_count += 1
                continue
            state, prepare_exec_started_at, prepare_exec_finished_at = result
            state.prepare_profile["engine_prepare_queue_admission_wait_ms"] = float(task.admission_wait_ms)
            state.prepare_profile["engine_gpu_prepare_queue_wait_ms"] = float(queue_wait_ms)
            state.prepare_profile["engine_gpu_prepare_batch_size"] = float(len(tasks))
            if task.engine_request_id not in [None, ""]:
                self.merge_request_state_profile(
                    str(task.engine_request_id),
                    {
                        "engine_prepare_queue_admission_wait_ms": float(task.admission_wait_ms),
                        "engine_gpu_prepare_queue_wait_ms": float(queue_wait_ms),
                        "engine_gpu_prepare_batch_size": float(len(tasks)),
                    },
                )
            self._notify_prepare_result(task, (state, prepare_exec_started_at, prepare_exec_finished_at))
            completed_count += 1
        self.prepare_queue_owner.mark_completed(completed_count)
        return True
