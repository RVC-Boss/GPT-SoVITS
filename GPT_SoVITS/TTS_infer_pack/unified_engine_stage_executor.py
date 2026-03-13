from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SFinishedItem
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import (
    EngineDecodeRuntimeOwner,
    EngineTaskQueueOwner,
    SchedulerFinalizeTask,
    SchedulerPendingJob,
)
from GPT_SoVITS.TTS_infer_pack.unified_engine_stage_decode import EngineDecodeStageMixin
from GPT_SoVITS.TTS_infer_pack.unified_engine_stage_dispatch import EngineDispatchStageMixin
from GPT_SoVITS.TTS_infer_pack.unified_engine_stage_finalize import EngineFinalizeStageMixin
from GPT_SoVITS.TTS_infer_pack.unified_engine_stage_futures import EngineStageFutureMixin
from GPT_SoVITS.TTS_infer_pack.unified_engine_stage_prepare import EnginePrepareStageMixin
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker import UnifiedSchedulerWorker


class EngineStageExecutor(
    EngineStageFutureMixin,
    EnginePrepareStageMixin,
    EngineFinalizeStageMixin,
    EngineDispatchStageMixin,
    EngineDecodeStageMixin,
):
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
        self.tts = tts
        self.scheduler_worker = scheduler_worker
        self.prepare_queue_owner = prepare_queue_owner
        self.prepare_text_queue_owner = prepare_text_queue_owner
        self.prepare_ref_spec_queue_owner = prepare_ref_spec_queue_owner
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
