from __future__ import annotations

import os
import threading
from typing import Callable, List

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import RuntimeStateCallbacks, SchedulerFinalizeTask, SchedulerJobRegistry
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker_completion import WorkerCompletionBridge
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker_decode import WorkerDecodeExecutor, WorkerDecodeLegacyShell, WorkerDecodeRuntimeTracker
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker_execution import WorkerExecutionMixin
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker_finalize import WorkerFinalizeExecutor
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker_prepare import WorkerPrepareExecutor
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker_runtime import WorkerRuntimeBookkeepingMixin
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker_submit import WorkerSubmitLifecycleMixin


class UnifiedSchedulerWorker(
    WorkerSubmitLifecycleMixin,
    WorkerRuntimeBookkeepingMixin,
    WorkerExecutionMixin,
):
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
