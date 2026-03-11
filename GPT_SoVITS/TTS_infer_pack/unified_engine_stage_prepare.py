from __future__ import annotations

import asyncio
import time
from typing import Any

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineGpuPrepareTask, EngineStatus


class EnginePrepareStageMixin:
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
