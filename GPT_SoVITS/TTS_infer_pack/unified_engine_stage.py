from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Optional

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SFinishedItem, T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import (
    EngineDecodeRuntimeOwner,
    EngineDispatchTask,
    EngineTaskQueueOwner,
    SchedulerFinalizeTask,
    SchedulerPendingJob,
)
from GPT_SoVITS.TTS_infer_pack.unified_engine_orchestration import EngineStageOrchestrator
from GPT_SoVITS.TTS_infer_pack.unified_engine_stage_executor import EngineStageExecutor
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker import UnifiedSchedulerWorker


class EngineStageCoordinator:
    def __init__(
        self,
        *,
        tts: TTS,
        scheduler_worker: UnifiedSchedulerWorker,
        prepare_queue_owner: EngineTaskQueueOwner,
        prepare_text_queue_owner: EngineTaskQueueOwner,
        prepare_ref_spec_queue_owner: EngineTaskQueueOwner,
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
        self.executor = EngineStageExecutor(
            tts=tts,
            scheduler_worker=scheduler_worker,
            prepare_queue_owner=prepare_queue_owner,
            prepare_text_queue_owner=prepare_text_queue_owner,
            prepare_ref_spec_queue_owner=prepare_ref_spec_queue_owner,
            finalize_queue_owner=finalize_queue_owner,
            dispatch_queue_owner=dispatch_queue_owner,
            decode_runtime_owner=decode_runtime_owner,
            update_request_state=update_request_state,
            merge_request_state_profile=merge_request_state_profile,
            fail_request_state=fail_request_state,
            get_engine_job=get_engine_job,
            register_engine_job=register_engine_job,
            fail_engine_jobs=fail_engine_jobs,
            complete_engine_job=complete_engine_job,
            add_engine_prefill_time=add_engine_prefill_time,
            add_engine_merge_time=add_engine_merge_time,
            add_engine_decode_time=add_engine_decode_time,
            enqueue_engine_finished_items=enqueue_engine_finished_items,
            snapshot_engine_dispatch_state=snapshot_engine_dispatch_state,
            snapshot_engine_decode_runtime_state=snapshot_engine_decode_runtime_state,
        )
        self.orchestrator = EngineStageOrchestrator(
            executor=self.executor,
            scheduler_worker=scheduler_worker,
            prepare_queue_owner=prepare_queue_owner,
            prepare_text_queue_owner=prepare_text_queue_owner,
            prepare_ref_spec_queue_owner=prepare_ref_spec_queue_owner,
            finalize_queue_owner=finalize_queue_owner,
            dispatch_queue_owner=dispatch_queue_owner,
            decode_runtime_owner=decode_runtime_owner,
            snapshot_engine_decode_runtime_state=snapshot_engine_decode_runtime_state,
        )

    def bind_arbiter(
        self,
        *,
        notify_arbiter: Callable[[], None],
        select_stage: Callable[[], tuple[str, str, Dict[str, Any], Dict[str, Any]]],
        mark_arbiter_tick: Callable[[str, str, bool], None],
        wait_arbiter: Callable[[], None],
    ) -> None:
        self.orchestrator.bind_arbiter(
            notify_arbiter=notify_arbiter,
            select_stage=select_stage,
            mark_arbiter_tick=mark_arbiter_tick,
            wait_arbiter=wait_arbiter,
        )

    async def prepare_state_via_engine_gpu_queue(
        self,
        *,
        spec,
        prepare_submit_at: float,
        engine_request_id: str | None,
    ) -> tuple[T2SRequestState, float, float]:
        return await self.executor.prepare_state_via_engine_gpu_queue(
            spec=spec,
            prepare_submit_at=prepare_submit_at,
            engine_request_id=engine_request_id,
        )

    def enqueue_worker_finished_for_finalize(self, tasks: List[SchedulerFinalizeTask]) -> None:
        self.executor.enqueue_worker_finished_for_finalize(tasks)

    def take_engine_finalize_batch_nonblocking(self) -> List[SchedulerFinalizeTask]:
        return self.executor.take_engine_finalize_batch_nonblocking()

    async def enqueue_prepared_state_for_dispatch(
        self,
        *,
        state: T2SRequestState,
        speed_factor: float,
        sample_steps: int,
        media_type: str,
        super_sampling: bool,
        prepare_wall_ms: float,
        prepare_profile_total_ms: float,
        done_loop: asyncio.AbstractEventLoop | None,
        done_future: asyncio.Future | None,
        engine_request_id: str | None,
        timeout_sec: float | None,
    ) -> EngineDispatchTask:
        return await self.executor.enqueue_prepared_state_for_dispatch(
            state=state,
            speed_factor=speed_factor,
            sample_steps=sample_steps,
            media_type=media_type,
            super_sampling=super_sampling,
            prepare_wall_ms=prepare_wall_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            done_loop=done_loop,
            done_future=done_future,
            engine_request_id=engine_request_id,
            timeout_sec=timeout_sec,
        )

    def peek_queue_age_ms(self, queue_name: str) -> float:
        return self.orchestrator.peek_queue_age_ms(queue_name)

    def has_pending_work(self) -> bool:
        return self.orchestrator.has_pending_work()

    def run_engine_prepare_once(self) -> bool:
        return self.executor.run_engine_prepare_once()

    def run_engine_prepare_audio_once(self) -> bool:
        return self.executor.run_engine_prepare_audio_once()

    def run_engine_prepare_text_once(self) -> bool:
        return self.executor.run_engine_prepare_text_once()

    def run_engine_prepare_ref_spec_once(self) -> bool:
        return self.executor.run_engine_prepare_ref_spec_once()

    def run_engine_finalize_once(self) -> bool:
        return self.executor.run_engine_finalize_once()

    def run_engine_dispatch_once(self, policy_snapshot: Dict[str, Any], worker_state: Dict[str, Any]) -> bool:
        return self.executor.run_engine_dispatch_once(policy_snapshot, worker_state)

    def run_engine_decode_runtime_once(self) -> bool:
        return self.executor.run_engine_decode_runtime_once()

    def run_engine_arbiter_loop(self) -> None:
        self.orchestrator.run_engine_arbiter_loop()
