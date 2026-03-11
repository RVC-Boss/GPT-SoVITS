from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

from GPT_SoVITS.TTS_infer_pack.unified_engine_api_direct import EngineApiDirectFlow
from GPT_SoVITS.TTS_infer_pack.unified_engine_api_profile import (
    aggregate_numeric_dicts,
    build_direct_scheduler_profile,
    build_direct_segment_trace,
    build_legacy_direct_profile,
    build_request_meta,
    build_scheduler_debug_batch_profile,
    build_scheduler_debug_request_profile,
    build_scheduler_submit_headers,
    build_scheduler_submit_profile,
    format_ms_header,
    sum_profile_field,
)
from GPT_SoVITS.TTS_infer_pack.unified_engine_api_request import (
    apply_default_reference,
    base_request_defaults,
    check_params,
    is_aux_ref_enabled,
    normalize_engine_request,
    normalize_lang,
    normalize_streaming_mode,
    select_direct_backend,
)
from GPT_SoVITS.TTS_infer_pack.unified_engine_api_scheduler import EngineApiSchedulerFlow
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SFinishedItem, T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import (
    DirectTTSExecution,
    NormalizedEngineRequest,
    SchedulerDebugExecution,
    SchedulerSubmitExecution,
)


class EngineApiFacade:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.direct_flow = EngineApiDirectFlow(self)
        self.scheduler_flow = EngineApiSchedulerFlow(self)

    @property
    def tts(self):
        return self.owner.tts

    @property
    def cut_method_names(self):
        return self.owner.cut_method_names

    @property
    def reference_registry(self):
        return self.owner.reference_registry

    @property
    def direct_tts_lock(self):
        return self.owner.direct_tts_lock

    @property
    def scheduler_worker(self):
        return self.owner.scheduler_worker

    def _register_request_state(
        self,
        request_id: str,
        api_mode: str,
        backend: str,
        media_type: str,
        response_streaming: bool,
        deadline_ts: float | None = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        return self.owner._register_request_state(
            request_id=request_id,
            api_mode=api_mode,
            backend=backend,
            media_type=media_type,
            response_streaming=response_streaming,
            deadline_ts=deadline_ts,
            meta=meta,
        )

    def _update_request_state(
        self,
        request_id: str,
        status: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.owner._update_request_state(request_id, status, extra)

    def _merge_request_state_profile(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.owner._merge_request_state_profile(request_id, extra)

    def _complete_request_state(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.owner._complete_request_state(request_id, extra)

    def _fail_request_state(self, request_id: str, error: str) -> None:
        self.owner._fail_request_state(request_id, error)

    async def _prepare_state_via_engine_gpu_queue(
        self,
        *,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
        engine_request_id: str | None,
    ) -> tuple[T2SRequestState, float, float]:
        return await self.owner._prepare_state_via_engine_gpu_queue(
            spec=spec,
            prepare_submit_at=prepare_submit_at,
            engine_request_id=engine_request_id,
        )

    async def _enqueue_prepared_state_for_dispatch(
        self,
        *,
        state: T2SRequestState,
        speed_factor: float,
        sample_steps: int,
        media_type: str,
        prepare_wall_ms: float,
        prepare_profile_total_ms: float,
        done_loop: asyncio.AbstractEventLoop | None,
        done_future: asyncio.Future | None,
        engine_request_id: str | None,
        timeout_sec: float | None,
    ):
        return await self.owner._enqueue_prepared_state_for_dispatch(
            state=state,
            speed_factor=speed_factor,
            sample_steps=sample_steps,
            media_type=media_type,
            prepare_wall_ms=prepare_wall_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            done_loop=done_loop,
            done_future=done_future,
            engine_request_id=engine_request_id,
            timeout_sec=timeout_sec,
        )

    def _collect_request_summaries(self, request_ids: Sequence[str]) -> List[Dict[str, Any]]:
        return self.owner.request_registry.collect_summaries(request_ids)

    def _has_active_request(self, request_id: str) -> bool:
        return self.owner.request_registry.has_active(request_id)

    @staticmethod
    def _build_request_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
        return build_request_meta(payload)

    @staticmethod
    def _sum_profile_field(items: Sequence[Dict[str, Any]], key: str) -> float:
        return sum_profile_field(items, key)

    def _build_direct_segment_trace(
        self,
        segment_texts: Sequence[str],
        prepare_profiles: Sequence[Dict[str, Any]],
        worker_profiles: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return build_direct_segment_trace(segment_texts, prepare_profiles, worker_profiles)

    def _build_direct_scheduler_profile(
        self,
        *,
        backend: str,
        request_start: float,
        response_ready_at: float,
        audio_bytes: int,
        sample_rate: int,
        segment_texts: Sequence[str],
        prepare_profiles: Sequence[Dict[str, Any]],
        worker_profiles: Sequence[Dict[str, Any]],
        pack_ms: float,
        response_overhead_ms: float,
    ) -> Dict[str, Any]:
        return build_direct_scheduler_profile(
            backend=backend,
            request_start=request_start,
            response_ready_at=response_ready_at,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            segment_texts=segment_texts,
            prepare_profiles=prepare_profiles,
            worker_profiles=worker_profiles,
            pack_ms=pack_ms,
            response_overhead_ms=response_overhead_ms,
        )

    def _build_legacy_direct_profile(
        self,
        *,
        backend: str,
        fallback_reason: str | None,
        request_start: float,
        finished_at: float,
        sample_rate: int | None = None,
        audio_bytes: int = 0,
        pack_ms: float = 0.0,
        chunk_count: int = 0,
        stream_total_bytes: int = 0,
        first_chunk_ms: float | None = None,
    ) -> Dict[str, Any]:
        return build_legacy_direct_profile(
            backend=backend,
            fallback_reason=fallback_reason,
            request_start=request_start,
            finished_at=finished_at,
            sample_rate=sample_rate,
            audio_bytes=audio_bytes,
            pack_ms=pack_ms,
            chunk_count=chunk_count,
            stream_total_bytes=stream_total_bytes,
            first_chunk_ms=first_chunk_ms,
        )

    def _build_scheduler_submit_profile(
        self,
        *,
        backend: str,
        request_start: float,
        response_ready_at: float,
        audio_bytes: int,
        sample_rate: int,
        prepare_spec_build_ms: float,
        prepare_wall_ms: float,
        prepare_executor_queue_ms: float,
        prepare_executor_run_ms: float,
        prepare_profile_total_ms: float,
        prepare_profile_wall_ms: float,
        prepare_other_ms: float,
        engine_policy_wait_ms: float,
        api_after_prepare_ms: float,
        api_wait_result_ms: float,
        pack_ms: float,
        response_overhead_ms: float,
        worker_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        return build_scheduler_submit_profile(
            backend=backend,
            request_start=request_start,
            response_ready_at=response_ready_at,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            prepare_spec_build_ms=prepare_spec_build_ms,
            prepare_wall_ms=prepare_wall_ms,
            prepare_executor_queue_ms=prepare_executor_queue_ms,
            prepare_executor_run_ms=prepare_executor_run_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            prepare_profile_wall_ms=prepare_profile_wall_ms,
            prepare_other_ms=prepare_other_ms,
            engine_policy_wait_ms=engine_policy_wait_ms,
            api_after_prepare_ms=api_after_prepare_ms,
            api_wait_result_ms=api_wait_result_ms,
            pack_ms=pack_ms,
            response_overhead_ms=response_overhead_ms,
            worker_profile=worker_profile,
        )

    @staticmethod
    def _format_ms_header(value: Any) -> str:
        return format_ms_header(value)

    def _build_scheduler_submit_headers(
        self,
        *,
        request_id: str,
        media_type: str,
        sample_rate: int,
        profile: Dict[str, Any],
    ) -> Dict[str, str]:
        return build_scheduler_submit_headers(
            request_id=request_id,
            media_type=media_type,
            sample_rate=sample_rate,
            profile=profile,
        )

    def _build_scheduler_debug_request_profile(
        self,
        *,
        state: T2SRequestState,
        item: T2SFinishedItem,
        batch_request_count: int,
        prepare_batch_wall_ms: float,
        decode_batch_wall_ms: float,
        batch_request_total_ms: float,
    ) -> Dict[str, Any]:
        return build_scheduler_debug_request_profile(
            state=state,
            item=item,
            batch_request_count=batch_request_count,
            prepare_batch_wall_ms=prepare_batch_wall_ms,
            decode_batch_wall_ms=decode_batch_wall_ms,
            batch_request_total_ms=batch_request_total_ms,
        )

    @staticmethod
    def _build_scheduler_debug_batch_profile(
        *,
        request_count: int,
        max_steps: int,
        prepare_batch_wall_ms: float,
        decode_batch_wall_ms: float,
        request_total_ms: float,
        finished_items: Sequence[T2SFinishedItem],
    ) -> Dict[str, Any]:
        return build_scheduler_debug_batch_profile(
            request_count=request_count,
            max_steps=max_steps,
            prepare_batch_wall_ms=prepare_batch_wall_ms,
            decode_batch_wall_ms=decode_batch_wall_ms,
            request_total_ms=request_total_ms,
            finished_items=finished_items,
        )

    def _normalize_lang(self, value: str | None) -> str | None:
        return normalize_lang(value)

    @staticmethod
    def _aggregate_numeric_dicts(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        return aggregate_numeric_dicts(items)

    def _apply_default_reference(self, req: dict) -> dict:
        return apply_default_reference(self.reference_registry, req)

    def check_params(self, req: dict) -> Optional[str]:
        return check_params(self.tts, self.cut_method_names, req)

    @staticmethod
    def _base_request_defaults() -> Dict[str, Any]:
        return base_request_defaults()

    def _normalize_engine_request(
        self,
        payload: dict | NormalizedEngineRequest,
        *,
        request_id: str | None = None,
        normalize_streaming: bool = False,
        error_prefix: str = "request 参数非法: ",
    ) -> NormalizedEngineRequest:
        return normalize_engine_request(
            tts=self.tts,
            cut_method_names=self.cut_method_names,
            reference_registry=self.reference_registry,
            payload=payload,
            request_id=request_id,
            normalize_streaming=normalize_streaming,
            error_prefix=error_prefix,
        )

    @staticmethod
    def _normalize_streaming_mode(req: dict) -> dict:
        return normalize_streaming_mode(req)

    @staticmethod
    def _is_aux_ref_enabled(aux_ref_audio_paths: List[str] | None) -> bool:
        return is_aux_ref_enabled(aux_ref_audio_paths)

    def _select_direct_backend(self, normalized: NormalizedEngineRequest) -> Tuple[str, str | None]:
        return select_direct_backend(normalized)

    def _iter_legacy_direct_tts_bytes(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> Generator[bytes, None, None]:
        yield from self.direct_flow._iter_legacy_direct_tts_bytes(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    def _should_use_scheduler_backend_for_direct(self, req: dict | NormalizedEngineRequest) -> bool:
        return self.direct_flow._should_use_scheduler_backend_for_direct(req)

    def _segment_direct_text(self, normalized: dict | NormalizedEngineRequest) -> List[str]:
        return self.direct_flow._segment_direct_text(normalized)

    def _build_segment_request(
        self,
        normalized: NormalizedEngineRequest,
        *,
        request_id: str,
        text: str,
    ) -> NormalizedEngineRequest:
        return self.direct_flow._build_segment_request(
            normalized,
            request_id=request_id,
            text=text,
        )

    async def _run_direct_tts_via_scheduler(self, normalized: NormalizedEngineRequest) -> DirectTTSExecution:
        return await self.direct_flow._run_direct_tts_via_scheduler(normalized)

    def _run_legacy_direct_tts_blocking(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        return self.direct_flow._run_legacy_direct_tts_blocking(
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
        return await self.direct_flow._run_direct_tts_via_legacy_backend(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    async def run_direct_tts_async(self, req: dict) -> DirectTTSExecution:
        return await self.direct_flow.run_direct_tts_async(req)

    def run_direct_tts(self, req: dict) -> DirectTTSExecution:
        return self.direct_flow.run_direct_tts(req)

    def _build_scheduler_request_specs(self, request_items: List[dict]) -> List[SchedulerRequestSpec]:
        return self.scheduler_flow._build_scheduler_request_specs(request_items)

    def _build_scheduler_submit_spec(self, payload: dict | NormalizedEngineRequest) -> SchedulerRequestSpec:
        return self.scheduler_flow._build_scheduler_submit_spec(payload)

    @staticmethod
    def _summarize_scheduler_states(states: List[T2SRequestState]) -> List[dict]:
        return EngineApiSchedulerFlow._summarize_scheduler_states(states)

    @staticmethod
    def _summarize_scheduler_finished(items: List[T2SFinishedItem]) -> List[dict]:
        return EngineApiSchedulerFlow._summarize_scheduler_finished(items)

    async def run_scheduler_debug(self, request_items: List[dict], max_steps: int, seed: int) -> SchedulerDebugExecution:
        return await self.scheduler_flow.run_scheduler_debug(request_items, max_steps, seed)

    async def run_scheduler_submit(self, payload: dict) -> SchedulerSubmitExecution:
        return await self.scheduler_flow.run_scheduler_submit(payload)
