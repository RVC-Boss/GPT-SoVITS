from __future__ import annotations

from typing import Any, Dict

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SActiveBatch
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineDecodeRuntimeOwner


class EngineRuntimeBridgeFacade:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    @property
    def engine_policy_arbiter(self):
        return self.owner.engine_policy_arbiter

    @staticmethod
    def _summarize_active_batch(active_batch: T2SActiveBatch | None) -> Dict[str, Any]:
        return EngineDecodeRuntimeOwner.summarize_active_batch(active_batch)

    def _snapshot_engine_arbiter_state(self) -> Dict[str, Any]:
        return self.engine_policy_arbiter.snapshot_state()

    def _notify_engine_arbiter(self) -> None:
        self.engine_policy_arbiter.notify()

    def _mark_arbiter_tick(self, *, stage: str, reason: str, policy_allowed: bool) -> None:
        self.engine_policy_arbiter.mark_tick(stage=stage, reason=reason, policy_allowed=policy_allowed)

    def _select_engine_stage(self) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        stage, reason, policy_snapshot, worker_state = self.engine_policy_arbiter.select_stage()
        self.owner.engine_dispatch_last_snapshot = dict(policy_snapshot)
        return stage, reason, policy_snapshot, worker_state
