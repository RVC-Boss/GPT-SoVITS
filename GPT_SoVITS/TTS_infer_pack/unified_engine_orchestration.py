from __future__ import annotations

from typing import Any, Callable, Dict

from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineDecodeRuntimeOwner, EngineTaskQueueOwner
from GPT_SoVITS.TTS_infer_pack.unified_engine_stage_executor import EngineStageExecutor
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker import UnifiedSchedulerWorker


class EngineStageOrchestrator:
    def __init__(
        self,
        *,
        executor: EngineStageExecutor,
        scheduler_worker: UnifiedSchedulerWorker,
        prepare_queue_owner: EngineTaskQueueOwner,
        prepare_text_queue_owner: EngineTaskQueueOwner,
        prepare_ref_spec_queue_owner: EngineTaskQueueOwner,
        finalize_queue_owner: EngineTaskQueueOwner,
        dispatch_queue_owner: EngineTaskQueueOwner,
        decode_runtime_owner: EngineDecodeRuntimeOwner,
        snapshot_engine_decode_runtime_state: Callable[[], Dict[str, Any]],
    ) -> None:
        self.executor = executor
        self.scheduler_worker = scheduler_worker
        self.prepare_queue_owner = prepare_queue_owner
        self.prepare_text_queue_owner = prepare_text_queue_owner
        self.prepare_ref_spec_queue_owner = prepare_ref_spec_queue_owner
        self.finalize_queue_owner = finalize_queue_owner
        self.dispatch_queue_owner = dispatch_queue_owner
        self.decode_runtime_owner = decode_runtime_owner
        self.snapshot_engine_decode_runtime_state = snapshot_engine_decode_runtime_state
        self._select_stage: Callable[[], tuple[str, str, Dict[str, Any], Dict[str, Any]]] | None = None
        self._mark_arbiter_tick: Callable[[str, str, bool], None] | None = None
        self._wait_arbiter: Callable[[], None] | None = None

    def bind_arbiter(
        self,
        *,
        notify_arbiter: Callable[[], None],
        select_stage: Callable[[], tuple[str, str, Dict[str, Any], Dict[str, Any]]],
        mark_arbiter_tick: Callable[[str, str, bool], None],
        wait_arbiter: Callable[[], None],
    ) -> None:
        self.executor.bind_notify_arbiter(notify_arbiter)
        self._select_stage = select_stage
        self._mark_arbiter_tick = mark_arbiter_tick
        self._wait_arbiter = wait_arbiter

    def peek_queue_age_ms(self, queue_name: str) -> float:
        if queue_name == "prepare":
            return max(
                self.prepare_queue_owner.peek_oldest_age_ms("enqueue_time"),
                self.prepare_text_queue_owner.peek_oldest_age_ms("enqueue_time"),
                self.prepare_ref_spec_queue_owner.peek_oldest_age_ms("enqueue_time"),
            )
        if queue_name == "prepare_audio":
            return self.prepare_queue_owner.peek_oldest_age_ms("enqueue_time")
        if queue_name == "prepare_text":
            return self.prepare_text_queue_owner.peek_oldest_age_ms("enqueue_time")
        if queue_name == "prepare_ref_spec":
            return self.prepare_ref_spec_queue_owner.peek_oldest_age_ms("enqueue_time")
        if queue_name == "finalize":
            return self.finalize_queue_owner.peek_oldest_age_ms("enqueued_time")
        if queue_name == "decode_runtime_pending":
            return self.decode_runtime_owner.pending_age_ms()
        return self.dispatch_queue_owner.peek_oldest_age_ms("enqueue_time")

    def has_pending_work(self) -> bool:
        if self.scheduler_worker.is_engine_decode_control_enabled():
            if self.decode_runtime_owner.has_pending_jobs():
                return True
        if self.scheduler_worker.is_engine_decode_control_enabled() and self.snapshot_engine_decode_runtime_state().get(
            "active_request_count", 0
        ) > 0:
            return True
        if self.prepare_queue_owner.has_items():
            return True
        if self.prepare_text_queue_owner.has_items():
            return True
        if self.prepare_ref_spec_queue_owner.has_items():
            return True
        if self.finalize_queue_owner.has_items():
            return True
        return self.dispatch_queue_owner.has_items()

    def run_engine_arbiter_loop(self) -> None:
        if self._select_stage is None or self._mark_arbiter_tick is None or self._wait_arbiter is None:
            raise RuntimeError("arbiter callbacks are not bound")
        while True:
            if not self.has_pending_work():
                self._mark_arbiter_tick("idle", "no_pending_work", True)
                self._wait_arbiter()
                continue
            stage, reason, policy_snapshot, worker_state = self._select_stage()
            policy_allowed = bool(policy_snapshot.get("allowed", True))
            executed = False
            if stage == "prepare":
                executed = self.executor.run_engine_prepare_once()
            elif stage == "prepare_audio":
                executed = self.executor.run_engine_prepare_audio_once()
            elif stage == "prepare_text":
                executed = self.executor.run_engine_prepare_text_once()
            elif stage == "prepare_ref_spec":
                executed = self.executor.run_engine_prepare_ref_spec_once()
            elif stage == "finalize":
                executed = self.executor.run_engine_finalize_once()
            elif stage == "decode_dispatch":
                executed = self.executor.run_engine_dispatch_once(policy_snapshot, worker_state)
            elif stage == "decode_runtime":
                executed = self.executor.run_engine_decode_runtime_once()
            if not executed:
                self._mark_arbiter_tick("idle", f"{stage}_not_ready", policy_allowed)
                self._wait_arbiter()
                continue
            self._mark_arbiter_tick(stage, reason, policy_allowed)
