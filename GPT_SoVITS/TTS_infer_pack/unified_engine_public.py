from __future__ import annotations

from GPT_SoVITS.TTS_infer_pack.unified_engine_components import DirectTTSExecution, SchedulerDebugExecution, SchedulerSubmitExecution


class EnginePublicInterface:
    PUBLIC_API_METHODS = (
        "run_direct_tts_async",
        "run_scheduler_submit",
        "run_scheduler_debug",
        "get_runtime_state",
        "set_refer_audio",
        "set_gpt_weights",
        "set_sovits_weights",
        "handle_control",
    )

    async def run_direct_tts_async(self, req: dict) -> DirectTTSExecution:
        return await self.api_facade.run_direct_tts_async(req)

    async def run_scheduler_debug(self, request_items: list[dict], max_steps: int, seed: int) -> SchedulerDebugExecution:
        return await self.api_facade.run_scheduler_debug(request_items, max_steps, seed)

    async def run_scheduler_submit(self, payload: dict) -> SchedulerSubmitExecution:
        return await self.api_facade.run_scheduler_submit(payload)

    def get_runtime_state(self) -> dict:
        return self.runtime_facade.get_runtime_state()

    def set_refer_audio(self, refer_audio_path: str | None) -> dict:
        return self.runtime_facade.set_refer_audio(refer_audio_path)

    def set_gpt_weights(self, weights_path: str) -> dict:
        return self.runtime_facade.set_gpt_weights(weights_path)

    def set_sovits_weights(self, weights_path: str) -> dict:
        return self.runtime_facade.set_sovits_weights(weights_path)

    def handle_control(self, command: str) -> None:
        self.runtime_facade.handle_control(command)


class EngineCompatInterface:
    COMPAT_API_METHODS = (
        "run_direct_tts",
        "get_scheduler_state",
    )

    def run_direct_tts(self, req: dict) -> DirectTTSExecution:
        return self.api_facade.run_direct_tts(req)

    def get_scheduler_state(self) -> dict:
        return self.runtime_facade.get_scheduler_state()
