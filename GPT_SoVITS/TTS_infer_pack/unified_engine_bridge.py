from __future__ import annotations

from typing import Any

from GPT_SoVITS.TTS_infer_pack.unified_engine_bridge_registry import EngineRegistryBridgeFacade
from GPT_SoVITS.TTS_infer_pack.unified_engine_bridge_runtime import EngineRuntimeBridgeFacade
from GPT_SoVITS.TTS_infer_pack.unified_engine_bridge_stage import EngineStageBridgeFacade


class EngineBridgeFacade:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.registry_bridge = EngineRegistryBridgeFacade(owner)
        self.stage_bridge = EngineStageBridgeFacade(owner)
        self.runtime_bridge = EngineRuntimeBridgeFacade(owner)

    def __getattr__(self, name: str) -> Any:
        for component in (self.registry_bridge, self.stage_bridge, self.runtime_bridge):
            if hasattr(component, name):
                return getattr(component, name)
        raise AttributeError(name)
