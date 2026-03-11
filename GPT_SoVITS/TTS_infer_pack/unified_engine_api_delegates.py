from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

from GPT_SoVITS.TTS_infer_pack.unified_engine_api import EngineApiFacade
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import DirectTTSExecution, NormalizedEngineRequest


class EngineApiDelegates:
    def _collect_request_summaries(self, request_ids: Sequence[str]) -> List[Dict[str, Any]]:
        return self.api_facade._collect_request_summaries(request_ids)

    def _has_active_request(self, request_id: str) -> bool:
        return self.api_facade._has_active_request(request_id)

    @staticmethod
    def _build_request_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
        return EngineApiFacade._build_request_meta(payload)

    @staticmethod
    def _sum_profile_field(items: Sequence[Dict[str, Any]], key: str) -> float:
        return EngineApiFacade._sum_profile_field(items, key)

    def _build_direct_segment_trace(
        self,
        segment_texts: Sequence[str],
        prepare_profiles: Sequence[Dict[str, Any]],
        worker_profiles: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self.api_facade._build_direct_segment_trace(segment_texts, prepare_profiles, worker_profiles)

    def _build_direct_scheduler_profile(self, **kwargs: Any) -> Dict[str, Any]:
        return self.api_facade._build_direct_scheduler_profile(**kwargs)

    def _build_legacy_direct_profile(self, **kwargs: Any) -> Dict[str, Any]:
        return self.api_facade._build_legacy_direct_profile(**kwargs)

    def _build_scheduler_submit_profile(self, **kwargs: Any) -> Dict[str, Any]:
        return self.api_facade._build_scheduler_submit_profile(**kwargs)

    @staticmethod
    def _format_ms_header(value: Any) -> str:
        return EngineApiFacade._format_ms_header(value)

    def _build_scheduler_submit_headers(
        self,
        *,
        request_id: str,
        media_type: str,
        sample_rate: int,
        profile: Dict[str, Any],
    ) -> Dict[str, str]:
        return self.api_facade._build_scheduler_submit_headers(
            request_id=request_id,
            media_type=media_type,
            sample_rate=sample_rate,
            profile=profile,
        )

    def _build_scheduler_debug_request_profile(self, **kwargs: Any) -> Dict[str, Any]:
        return self.api_facade._build_scheduler_debug_request_profile(**kwargs)

    @staticmethod
    def _build_scheduler_debug_batch_profile(**kwargs: Any) -> Dict[str, Any]:
        return EngineApiFacade._build_scheduler_debug_batch_profile(**kwargs)

    def _normalize_lang(self, value: str | None) -> str | None:
        return self.api_facade._normalize_lang(value)

    @staticmethod
    def _aggregate_numeric_dicts(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        return EngineApiFacade._aggregate_numeric_dicts(items)

    def _apply_default_reference(self, req: dict) -> dict:
        return self.api_facade._apply_default_reference(req)

    def check_params(self, req: dict) -> Optional[str]:
        return self.api_facade.check_params(req)

    @staticmethod
    def _base_request_defaults() -> Dict[str, Any]:
        return EngineApiFacade._base_request_defaults()

    def _normalize_engine_request(
        self,
        payload: dict | NormalizedEngineRequest,
        *,
        request_id: str | None = None,
        normalize_streaming: bool = False,
        error_prefix: str = "request 参数非法: ",
    ) -> NormalizedEngineRequest:
        return self.api_facade._normalize_engine_request(
            payload,
            request_id=request_id,
            normalize_streaming=normalize_streaming,
            error_prefix=error_prefix,
        )

    @staticmethod
    def _normalize_streaming_mode(req: dict) -> dict:
        return EngineApiFacade._normalize_streaming_mode(req)

    @staticmethod
    def _is_aux_ref_enabled(aux_ref_audio_paths: List[str] | None) -> bool:
        return EngineApiFacade._is_aux_ref_enabled(aux_ref_audio_paths)

    def _select_direct_backend(self, normalized: NormalizedEngineRequest) -> Tuple[str, str | None]:
        return self.api_facade._select_direct_backend(normalized)

    def _iter_legacy_direct_tts_bytes(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> Generator[bytes, None, None]:
        return self.api_facade._iter_legacy_direct_tts_bytes(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    def _should_use_scheduler_backend_for_direct(self, req: dict | NormalizedEngineRequest) -> bool:
        return self.api_facade._should_use_scheduler_backend_for_direct(req)

    def _segment_direct_text(self, normalized: dict | NormalizedEngineRequest) -> List[str]:
        return self.api_facade._segment_direct_text(normalized)

    def _build_segment_request(
        self,
        normalized: NormalizedEngineRequest,
        *,
        request_id: str,
        text: str,
    ) -> NormalizedEngineRequest:
        return self.api_facade._build_segment_request(normalized, request_id=request_id, text=text)

    async def _run_direct_tts_via_scheduler(self, normalized: NormalizedEngineRequest) -> DirectTTSExecution:
        return await self.api_facade._run_direct_tts_via_scheduler(normalized)

    def _run_legacy_direct_tts_blocking(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        return self.api_facade._run_legacy_direct_tts_blocking(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    async def _run_direct_tts_via_legacy_backend(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        return await self.api_facade._run_direct_tts_via_legacy_backend(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )
