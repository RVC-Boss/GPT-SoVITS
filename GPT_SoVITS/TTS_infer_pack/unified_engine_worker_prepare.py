from __future__ import annotations

import asyncio
import os
import time
from typing import Callable, Dict, List

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import PrepareCoordinator, PreparedCpuStage
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SRequestState


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

    def get_batch_policy(self) -> Dict[str, int]:
        return {
            "prepare_batch_max_items": max(1, int(os.environ.get("GPTSOVITS_ENGINE_PREPARE_BATCH_MAX_ITEMS", 8))),
        }

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

    async def prepare_gpu_stages_profiled_async(
        self,
        cpu_stages: List[PreparedCpuStage],
    ) -> List[tuple[T2SRequestState, float, float] | Exception]:
        try:
            return list(
                await asyncio.gather(
                    *[self.coordinator.prepare_gpu_stage_profiled_async(cpu_stage) for cpu_stage in cpu_stages],
                    return_exceptions=True,
                )
            )
        finally:
            self._notify_state_change()
