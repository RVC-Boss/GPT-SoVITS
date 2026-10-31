"""Compatibility launcher for the GPT-SoVITS HTTP inference service.

Start: python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
Usage and API reference: README.md and docs/api.md.
"""

import sys
import types
from pathlib import Path


# GPU-free compatibility tests and downstream launchers may inject the historic
# GPT_SoVITS inference modules while loading this file directly. Keep the real
# package path available so the new service package can coexist with those
# injected modules.
_package = sys.modules.get("GPT_SoVITS")
_package_path = str(Path(__file__).resolve().parent / "GPT_SoVITS")
if _package is not None and hasattr(_package, "__path__") and _package_path not in _package.__path__:
    _package.__path__.append(_package_path)

from GPT_SoVITS.service.app import (
    APP,
    DEFAULT_CONFIG_PATH,
    TTS,
    TTS_Config,
    create_app,
    lifespan,
    state as _service_state,
)
from GPT_SoVITS.service.audio_response import (
    pack_aac,
    pack_audio,
    pack_ogg,
    pack_raw,
    pack_wav,
    wave_header_chunk,
)
from GPT_SoVITS.service.cli import main
from GPT_SoVITS.service.concurrency import (
    STREAM_END as _STREAM_END,
    await_shielded as _await_shielded,
    next_or_end as _next_or_end,
    next_tts_chunk as _next_tts_chunk,
    run_in_threadpool_shielded as _run_in_threadpool_shielded,
)
from GPT_SoVITS.service.routes import (
    check_params,
    cut_method_names,
    set_gpt_weights,
    set_refer_aduio,
    set_sovits_weights,
    tts_get_endpoint,
    tts_handle,
    tts_post_endpoint,
)
from GPT_SoVITS.service.schemas import TTS_Request


# Keep the legacy names visible in the module namespace as well as dynamically
# synchronized below. This preserves introspection and ``from api_v2 import *``.
tts_config = _service_state.tts_config
tts_pipeline = _service_state.tts_pipeline
_tts_pipeline_lock = _service_state.pipeline_lock


class _CompatibilityModule(types.ModuleType):
    """Keep mutable legacy globals synchronized with explicit service state."""

    _state_attributes = {
        "tts_config": "tts_config",
        "tts_pipeline": "tts_pipeline",
        "_tts_pipeline_lock": "pipeline_lock",
    }

    def __getattribute__(self, name):
        state_attributes = types.ModuleType.__getattribute__(self, "_state_attributes")
        if name in state_attributes:
            state = types.ModuleType.__getattribute__(self, "_service_state")
            return getattr(state, state_attributes[name])
        return types.ModuleType.__getattribute__(self, name)

    def __setattr__(self, name, value):
        state_attributes = types.ModuleType.__getattribute__(self, "_state_attributes")
        if name in state_attributes:
            state = types.ModuleType.__getattribute__(self, "_service_state")
            setattr(state, state_attributes[name], value)
            return
        types.ModuleType.__setattr__(self, name, value)


sys.modules[__name__].__class__ = _CompatibilityModule


if __name__ == "__main__":
    main()
