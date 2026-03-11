from __future__ import annotations

import os
import signal
import sys
from typing import Any, Dict, Optional


class EngineRuntimeFacade:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    @property
    def tts(self):
        return self.owner.tts

    @property
    def reference_registry(self):
        return self.owner.reference_registry

    @property
    def model_registry(self):
        return self.owner.model_registry

    @property
    def scheduler_worker(self):
        return self.owner.scheduler_worker

    @property
    def engine_decode_runtime_owner(self):
        return self.owner.engine_decode_runtime_owner

    @property
    def engine_policy_arbiter(self):
        return self.owner.engine_policy_arbiter

    @property
    def management_lock(self):
        return self.owner.management_lock

    @property
    def direct_tts_lock(self):
        return self.owner.direct_tts_lock

    @property
    def control_callbacks(self):
        return self.owner.control_callbacks

    @staticmethod
    def _safe_component_snapshot(component: Any) -> Dict[str, Any] | None:
        if component is None or not hasattr(component, "snapshot"):
            return None
        try:
            return dict(component.snapshot())
        except Exception:
            return None

    def _build_stage_counters(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.engine_policy_arbiter.build_stage_counters(request_registry, worker_state)

    def _build_engine_policy_snapshot(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.engine_policy_arbiter.build_policy_snapshot(request_registry, worker_state)

    def _build_stage_summary(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        counters = self._build_stage_counters(request_registry, worker_state)
        bert_worker_state = self._safe_component_snapshot(getattr(self.tts, "prepare_bert_batch_worker", None))
        ref_semantic_worker_state = self._safe_component_snapshot(getattr(self.tts, "prepare_ref_semantic_batch_worker", None))
        text_preprocessor_state = self._safe_component_snapshot(getattr(self.tts, "text_preprocessor", None))

        return {
            **counters,
            "engine_drained": bool(self.owner._is_engine_drained()),
            "admission_config": {
                "decode_backlog_max": int(worker_state.get("decode_backlog_max", 0)),
                "finalize_pending_max": int(worker_state.get("finalize_pending_max", 0)),
            },
            "engine_policy": self._build_engine_policy_snapshot(request_registry, worker_state),
            "engine_arbiter_state": self.owner._snapshot_engine_arbiter_state(),
            "engine_decode_runtime_state": self.owner._snapshot_engine_decode_runtime_state(),
            "engine_job_registry": self.owner._snapshot_engine_job_registry(),
            "engine_active_batch_state": self.engine_decode_runtime_owner.active_batch_summary(),
            "engine_prepare_state": self.owner._snapshot_engine_prepare_state(),
            "engine_finalize_state": self.owner._snapshot_engine_finalize_state(),
            "engine_dispatcher_state": self.owner._snapshot_engine_dispatch_state(),
            "active_batch": dict(worker_state.get("active_batch") or {}),
            "prepare_state": dict(worker_state.get("prepare_state") or {}),
            "bert_batch_worker_state": bert_worker_state,
            "ref_semantic_worker_state": ref_semantic_worker_state,
            "text_preprocessor_state": text_preprocessor_state,
        }

    def get_scheduler_state(self) -> dict:
        return self.scheduler_worker.snapshot()

    def get_runtime_state(self) -> dict:
        model_state = self.model_registry.snapshot()
        default_ref = self.reference_registry.get_default()
        scheduler_state = self.get_scheduler_state()
        request_registry = self.owner._snapshot_request_registry()
        engine_policy = self._build_engine_policy_snapshot(request_registry, scheduler_state)
        engine_arbiter_state = self.owner._snapshot_engine_arbiter_state()
        engine_decode_runtime_state = self.owner._snapshot_engine_decode_runtime_state()
        engine_job_registry = self.owner._snapshot_engine_job_registry()
        engine_prepare_state = self.owner._snapshot_engine_prepare_state()
        engine_finalize_state = self.owner._snapshot_engine_finalize_state()
        engine_dispatcher_state = self.owner._snapshot_engine_dispatch_state()
        engine_drained = self.owner._is_engine_drained()
        return {
            "message": "success",
            "default_reference": {
                "ref_audio_path": default_ref.ref_audio_path,
                "updated_at": default_ref.updated_at,
            },
            "model_registry": {
                "generation": model_state.generation,
                "t2s_generation": model_state.t2s_generation,
                "vits_generation": model_state.vits_generation,
                "t2s_weights_path": model_state.t2s_weights_path,
                "vits_weights_path": model_state.vits_weights_path,
                "updated_at": model_state.updated_at,
            },
            "worker_state": scheduler_state,
            "engine_policy": engine_policy,
            "engine_arbiter_state": engine_arbiter_state,
            "engine_decode_runtime_state": engine_decode_runtime_state,
            "engine_job_registry": engine_job_registry,
            "engine_active_batch_state": self.engine_decode_runtime_owner.active_batch_summary(),
            "engine_prepare_state": engine_prepare_state,
            "engine_finalize_state": engine_finalize_state,
            "engine_dispatcher_state": engine_dispatcher_state,
            "engine_drained": bool(engine_drained),
            "request_registry": request_registry,
            "stage_summary": self._build_stage_summary(request_registry, scheduler_state),
        }

    def _wait_for_safe_reload(self, timeout_sec: float = 300.0) -> None:
        if not self.scheduler_worker.wait_until_idle(timeout_sec=timeout_sec):
            raise TimeoutError("scheduler worker did not drain before model reload")

    def set_refer_audio(self, refer_audio_path: str | None) -> dict:
        if refer_audio_path in [None, ""]:
            state = self.reference_registry.clear()
            return {"message": "success", "default_ref_audio_path": state.ref_audio_path}
        if not os.path.exists(str(refer_audio_path)):
            raise FileNotFoundError(f"{refer_audio_path} not exists")
        with self.management_lock:
            with self.direct_tts_lock:
                self.tts.set_ref_audio(str(refer_audio_path))
            state = self.reference_registry.set_default(str(refer_audio_path))
        return {"message": "success", "default_ref_audio_path": state.ref_audio_path}

    def set_gpt_weights(self, weights_path: str) -> dict:
        if weights_path in ["", None]:
            raise ValueError("gpt weight path is required")
        with self.management_lock:
            self._wait_for_safe_reload()
            with self.direct_tts_lock:
                self.tts.init_t2s_weights(weights_path)
                self.tts.refresh_runtime_components()
            state = self.model_registry.mark_t2s_reload(str(weights_path))
        return {"message": "success", "t2s_generation": state.t2s_generation, "generation": state.generation}

    def set_sovits_weights(self, weights_path: str) -> dict:
        if weights_path in ["", None]:
            raise ValueError("sovits weight path is required")
        with self.management_lock:
            self._wait_for_safe_reload()
            with self.direct_tts_lock:
                self.tts.init_vits_weights(weights_path)
                self.tts.refresh_runtime_components()
            state = self.model_registry.mark_vits_reload(str(weights_path))
        return {"message": "success", "vits_generation": state.vits_generation, "generation": state.generation}

    def handle_control(self, command: str) -> None:
        if command == "restart":
            if self.control_callbacks.restart is None:
                os.execl(sys.executable, sys.executable, *sys.argv)
            self.control_callbacks.restart()
            return
        if command == "exit":
            if self.control_callbacks.exit is None:
                os.kill(os.getpid(), signal.SIGTERM)
                return
            self.control_callbacks.exit()
            return
        raise ValueError(f"unsupported command: {command}")
