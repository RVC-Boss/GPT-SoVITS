"""FastAPI application assembly and inference process lifecycle."""

import asyncio
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI

if "tools.portable_runtime" in sys.modules:
    from tools.portable_runtime import activate as activate_portable_runtime
else:
    try:
        from GPT_SoVITS.runtime.portable import activate as activate_portable_runtime
    except ImportError:
        from tools.portable_runtime import activate as activate_portable_runtime

activate_portable_runtime(change_cwd=True)

from GPT_SoVITS.inference import TTS, TTS_Config

from .concurrency import run_in_threadpool_shielded


DEFAULT_CONFIG_PATH = "GPT_SoVITS/configs/tts_infer.yaml"


@dataclass
class ServiceState:
    """Mutable state shared by routes in this single-worker application."""

    tts_config: Any
    tts_pipeline: Any = None
    pipeline_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Configuration construction remains import-time compatible; model construction
# remains deferred until the ASGI lifespan starts.
state = ServiceState(tts_config=TTS_Config(DEFAULT_CONFIG_PATH))


@asynccontextmanager
async def lifespan(_app):
    if state.tts_pipeline is None:
        print(state.tts_config)
        state.tts_pipeline = await run_in_threadpool_shielded(TTS, state.tts_config)
    try:
        yield
    finally:
        pipeline = state.tts_pipeline
        for name in ("streaming_accel", "t2s_accel"):
            accelerator = getattr(pipeline, name, None)
            if accelerator is not None:
                await run_in_threadpool_shielded(accelerator.close)
        state.tts_pipeline = None


def create_app() -> FastAPI:
    """Assemble the stable inference-only HTTP surface."""

    application = FastAPI(
        title="GPT-SoVITS 推理 API",
        description="使用已加载的 GPT/SoVITS 模型合成语音。模型和参考音频路径均由服务端读取；热切换仅作用于当前进程。",
        lifespan=lifespan,
    )
    from .routes import register_routes

    register_routes(application)
    return application


APP = create_app()
