from __future__ import annotations

from typing import Any, Dict

from GPT_SoVITS.TTS_infer_pack.unified_engine_runtime import EngineRuntimeFacade


class EngineRuntimeDelegates:
    @staticmethod
    def _safe_component_snapshot(component: Any) -> Dict[str, Any] | None:
        return EngineRuntimeFacade._safe_component_snapshot(component)

    def _build_stage_counters(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.runtime_facade._build_stage_counters(request_registry, worker_state)

    def _build_engine_policy_snapshot(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.runtime_facade._build_engine_policy_snapshot(request_registry, worker_state)

    async def _wait_for_engine_policy_admission(
        self,
        *,
        request_id: str | None,
        timeout_sec: float | None,
    ) -> tuple[float, Dict[str, Any]]:
        return await self.engine_policy_arbiter.wait_for_policy_admission(
            request_id=request_id,
            timeout_sec=timeout_sec,
        )

    def _build_stage_summary(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.runtime_facade._build_stage_summary(request_registry, worker_state)

    def _wait_for_safe_reload(self, timeout_sec: float = 300.0) -> None:
        self.runtime_facade._wait_for_safe_reload(timeout_sec=timeout_sec)
