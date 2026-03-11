from __future__ import annotations

import os
from typing import Sequence

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.unified_engine_builder import EngineCompositionBuilder
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import RuntimeControlCallbacks
from GPT_SoVITS.TTS_infer_pack.unified_engine_delegates import EngineApiDelegates, EngineBridgeDelegates, EngineRuntimeDelegates
from GPT_SoVITS.TTS_infer_pack.unified_engine_public import EngineCompatInterface, EnginePublicInterface


class UnifiedTTSEngine(EnginePublicInterface, EngineCompatInterface, EngineBridgeDelegates, EngineApiDelegates, EngineRuntimeDelegates):
    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value in [None, ""]:
            return int(default)
        return int(value)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value in [None, ""]:
            return float(default)
        return float(value)

    def __init__(
        self,
        tts: TTS,
        cut_method_names: Sequence[str],
        control_callbacks: RuntimeControlCallbacks | None = None,
        max_steps: int = 1500,
        micro_batch_wait_ms: int = 5,
    ) -> None:
        self.tts = tts
        self.cut_method_names = set(cut_method_names)
        self.control_callbacks = control_callbacks or RuntimeControlCallbacks()
        EngineCompositionBuilder(self).build(max_steps=max_steps, micro_batch_wait_ms=micro_batch_wait_ms)
