from __future__ import annotations

import asyncio
import time
import uuid
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional

import numpy as np

from GPT_SoVITS.TTS_infer_pack.unified_engine_audio import pack_audio, wave_header_chunk
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import DirectTTSExecution, EngineStatus, NormalizedEngineRequest, SchedulerPendingJob


class EngineApiDirectFlow:
    def __init__(self, api: Any) -> None:
        self.api = api

    def _iter_legacy_direct_tts_bytes(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> Generator[bytes, None, None]:
        payload = normalized.to_payload()
        media_type = normalized.media_type
        request_id = normalized.request_id
        request_start = time.perf_counter()
        chunk_count = 0
        stream_total_bytes = 0
        first_chunk_ms: float | None = None
        self.api._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        try:
            with self.api.direct_tts_lock:
                tts_generator = self.api.tts.run(payload)
                first_chunk = True
                current_media_type = media_type
                for sr, chunk in tts_generator:
                    if first_chunk:
                        first_chunk_ms = max(0.0, (time.perf_counter() - request_start) * 1000.0)
                        self.api._update_request_state(
                            request_id,
                            EngineStatus.STREAMING,
                            {
                                "backend": backend,
                                "backend_mode": backend,
                                "fallback_reason": fallback_reason,
                                "sample_rate": int(sr),
                            },
                        )
                    if first_chunk and media_type == "wav":
                        header = wave_header_chunk(sample_rate=sr)
                        chunk_count += 1
                        stream_total_bytes += len(header)
                        yield header
                        current_media_type = "raw"
                        first_chunk = False
                    elif first_chunk:
                        first_chunk = False
                    packed_chunk = pack_audio(BytesIO(), chunk, sr, current_media_type).getvalue()
                    chunk_count += 1
                    stream_total_bytes += len(packed_chunk)
                    yield packed_chunk
        except Exception as exc:
            self.api._fail_request_state(request_id, str(exc))
            raise
        self.api._complete_request_state(
            request_id,
            dict(
                self.api._build_legacy_direct_profile(
                    backend=backend,
                    fallback_reason=fallback_reason,
                    request_start=request_start,
                    finished_at=time.perf_counter(),
                    audio_bytes=stream_total_bytes,
                    chunk_count=chunk_count,
                    stream_total_bytes=stream_total_bytes,
                    first_chunk_ms=first_chunk_ms,
                ),
                streaming_completed=True,
            ),
        )

    def _should_use_scheduler_backend_for_direct(self, req: dict | NormalizedEngineRequest) -> bool:
        if isinstance(req, NormalizedEngineRequest):
            normalized = req
        else:
            normalized = self.api._normalize_engine_request(
                req,
                request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
                normalize_streaming=True,
            )
        backend, _ = self.api._select_direct_backend(normalized)
        return backend == "scheduler_v1_direct"

    def _segment_direct_text(self, normalized: dict | NormalizedEngineRequest) -> List[str]:
        payload = normalized.to_payload() if isinstance(normalized, NormalizedEngineRequest) else normalized
        return self.api.tts.text_preprocessor.pre_seg_text(
            str(payload["text"]),
            str(payload["text_lang"]),
            str(payload.get("text_split_method", "cut5")),
        )

    def _build_segment_request(
        self,
        normalized: NormalizedEngineRequest,
        *,
        request_id: str,
        text: str,
    ) -> NormalizedEngineRequest:
        payload = normalized.to_payload()
        payload["request_id"] = request_id
        payload["text"] = text
        payload["streaming_mode"] = False
        payload["return_fragment"] = False
        payload["fixed_length_chunk"] = False
        payload["response_streaming"] = False
        return self.api._normalize_engine_request(payload, error_prefix="segment request 参数非法: ")

    async def _run_direct_tts_via_scheduler(self, normalized: NormalizedEngineRequest) -> DirectTTSExecution:
        request_start = time.perf_counter()
        request_id = normalized.request_id
        media_type = normalized.media_type
        segment_texts = self._segment_direct_text(normalized)
        if not segment_texts:
            raise ValueError("text preprocessing returned no valid segments")
        self.api._update_request_state(
            request_id,
            EngineStatus.CPU_PREPARING,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct", "segment_count": len(segment_texts)},
        )
        segment_specs = []
        for segment_index, segment_text in enumerate(segment_texts):
            segment_request = self._build_segment_request(
                normalized,
                request_id=f"{request_id}_seg_{segment_index:03d}",
                text=segment_text,
            )
            segment_specs.append(self.api._build_scheduler_submit_spec(segment_request))

        prepared_items = await asyncio.gather(
            *[
                self.api._prepare_state_via_engine_gpu_queue(
                    spec=spec,
                    prepare_submit_at=time.perf_counter(),
                    engine_request_id=None,
                )
                for spec in segment_specs
            ]
        )
        prepare_profiles: List[Dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        done_futures: List[asyncio.Future] = []
        self.api._update_request_state(
            request_id,
            EngineStatus.READY_FOR_PREFILL,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct", "segment_count": len(segment_specs)},
        )
        for spec, (state, prepare_exec_started_at, prepare_exec_finished_at) in zip(segment_specs, prepared_items):
            prepare_wall_ms = max(0.0, (prepare_exec_finished_at - prepare_exec_started_at) * 1000.0)
            prepare_profile_total_ms = float(state.prepare_profile.get("wall_total_ms", prepare_wall_ms))
            prepare_profiles.append(
                {
                    "request_id": spec.request_id,
                    "prepare_wall_ms": prepare_wall_ms,
                    "prepare_profile_total_ms": prepare_profile_total_ms,
                    "prepare_profile": dict(state.prepare_profile),
                }
            )
            done_future = loop.create_future()
            done_futures.append(done_future)
            await self.api._enqueue_prepared_state_for_dispatch(
                state=state,
                speed_factor=float(normalized.speed_factor),
                sample_steps=int(normalized.sample_steps),
                media_type=media_type,
                prepare_wall_ms=prepare_wall_ms,
                prepare_profile_total_ms=prepare_profile_total_ms,
                done_loop=loop,
                done_future=done_future,
                engine_request_id=None,
                timeout_sec=normalized.timeout_sec,
            )
        self.api._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct"},
        )
        timeout_sec = float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0)
        jobs: List[SchedulerPendingJob] = list(await asyncio.wait_for(asyncio.gather(*done_futures), timeout=timeout_sec))
        for profile_item, job in zip(prepare_profiles, jobs):
            profile_item["engine_policy_wait_ms"] = float(job.engine_policy_wait_ms)
            profile_item["engine_dispatch_wait_ms"] = float(job.engine_dispatch_wait_ms)
        self.api._merge_request_state_profile(
            request_id,
            {
                "engine_policy_wait_ms": sum(float(job.engine_policy_wait_ms) for job in jobs),
                "engine_dispatch_wait_ms": sum(float(job.engine_dispatch_wait_ms) for job in jobs),
                "prepare_aggregate": self.api._aggregate_numeric_dicts([item["prepare_profile"] for item in prepare_profiles]),
            },
        )

        sample_rate: int | None = None
        audio_parts: List[np.ndarray] = []
        worker_profiles: List[Dict[str, Any]] = []
        fragment_interval = float(normalized.fragment_interval)
        silence_chunk: Optional[np.ndarray] = None
        for job in jobs:
            if job.error is not None:
                raise RuntimeError(job.error)
            if job.audio_data is None or job.sample_rate is None or job.result is None:
                raise RuntimeError(f"{job.request_id} finished without audio result")
            if sample_rate is None:
                sample_rate = int(job.sample_rate)
                silence_samples = int(fragment_interval * float(sample_rate))
                if silence_samples > 0:
                    silence_chunk = np.zeros(silence_samples, dtype=np.int16)
            elif int(job.sample_rate) != sample_rate:
                raise RuntimeError("segment sample rate mismatch")
            audio_parts.append(job.audio_data)
            if silence_chunk is not None:
                audio_parts.append(silence_chunk.copy())
            worker_profiles.append(dict(job.result))
        if sample_rate is None or not audio_parts:
            raise RuntimeError("direct scheduler backend produced no audio")
        self.api._update_request_state(
            request_id,
            EngineStatus.FINALIZING,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct"},
        )
        merged_audio = np.concatenate(audio_parts, axis=0)
        pack_start = time.perf_counter()
        audio_bytes = pack_audio(BytesIO(), merged_audio, sample_rate, media_type).getvalue()
        pack_ms = max(0.0, (time.perf_counter() - pack_start) * 1000.0)
        direct_profile = self.api._build_direct_scheduler_profile(
            backend="scheduler_v1_direct",
            request_start=request_start,
            response_ready_at=time.perf_counter(),
            audio_bytes=len(audio_bytes),
            sample_rate=int(sample_rate),
            segment_texts=segment_texts,
            prepare_profiles=prepare_profiles,
            worker_profiles=worker_profiles,
            pack_ms=pack_ms,
            response_overhead_ms=0.0,
        )
        self.api._complete_request_state(
            request_id,
            dict(direct_profile, streaming_completed=False),
        )
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=audio_bytes,
            request_id=request_id,
        )

    def _run_legacy_direct_tts_blocking(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        normalized_payload = normalized.to_payload()
        request_id = normalized.request_id
        media_type = normalized.media_type
        request_start = time.perf_counter()
        self.api._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        with self.api.direct_tts_lock:
            tts_generator = self.api.tts.run(normalized_payload)
            try:
                sr, audio_data = next(tts_generator)
            except Exception as exc:
                self.api._fail_request_state(request_id, str(exc))
                raise
        self.api._update_request_state(
            request_id,
            EngineStatus.FINALIZING,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        pack_start = time.perf_counter()
        packed_audio = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
        pack_ms = max(0.0, (time.perf_counter() - pack_start) * 1000.0)
        self.api._complete_request_state(
            request_id,
            dict(
                self.api._build_legacy_direct_profile(
                    backend=backend,
                    fallback_reason=fallback_reason,
                    request_start=request_start,
                    finished_at=time.perf_counter(),
                    sample_rate=int(sr),
                    audio_bytes=len(packed_audio),
                    pack_ms=pack_ms,
                ),
                streaming_completed=False,
            ),
        )
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=packed_audio,
            request_id=request_id,
        )

    async def _run_direct_tts_via_legacy_backend(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        if normalized.response_streaming:
            return DirectTTSExecution(
                media_type=normalized.media_type,
                streaming=True,
                audio_generator=self._iter_legacy_direct_tts_bytes(
                    normalized,
                    backend=backend,
                    fallback_reason=fallback_reason,
                ),
                request_id=normalized.request_id,
            )
        return await asyncio.to_thread(
            self._run_legacy_direct_tts_blocking,
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    async def run_direct_tts_async(self, req: dict) -> DirectTTSExecution:
        normalized = self.api._normalize_engine_request(
            req,
            request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
            normalize_streaming=True,
            error_prefix="",
        )
        request_id = normalized.request_id
        media_type = normalized.media_type
        backend, fallback_reason = self.api._select_direct_backend(normalized)
        self.api._register_request_state(
            request_id=request_id,
            api_mode="tts",
            backend=backend,
            media_type=media_type,
            response_streaming=bool(normalized.response_streaming),
            deadline_ts=(time.perf_counter() + float(normalized.timeout_sec) if normalized.timeout_sec is not None else None),
            meta=self.api._build_request_meta(normalized.to_payload()),
        )
        self.api._update_request_state(
            request_id,
            EngineStatus.VALIDATED,
            {
                "request_source": "direct_tts",
                "selected_backend": backend,
                "fallback_reason": fallback_reason,
            },
        )
        if backend == "scheduler_v1_direct":
            try:
                return await self._run_direct_tts_via_scheduler(normalized)
            except Exception as exc:
                self.api._fail_request_state(request_id, str(exc))
                raise
        return await self._run_direct_tts_via_legacy_backend(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    def run_direct_tts(self, req: dict) -> DirectTTSExecution:
        normalized = self.api._normalize_engine_request(
            req,
            request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
            normalize_streaming=True,
            error_prefix="",
        )
        request_id = normalized.request_id
        media_type = normalized.media_type
        backend, fallback_reason = self.api._select_direct_backend(normalized)
        if not self.api._has_active_request(request_id):
            self.api._register_request_state(
                request_id=request_id,
                api_mode="tts",
                backend=backend,
                media_type=media_type,
                response_streaming=bool(normalized.response_streaming),
                meta=self.api._build_request_meta(normalized.to_payload()),
            )
        self.api._update_request_state(
            request_id,
            EngineStatus.VALIDATED,
            {
                "request_source": "direct_tts",
                "selected_backend": backend,
                "fallback_reason": fallback_reason,
            },
        )
        if backend != "scheduler_v1_direct":
            if normalized.response_streaming:
                return DirectTTSExecution(
                    media_type=media_type,
                    streaming=True,
                    audio_generator=self._iter_legacy_direct_tts_bytes(
                        normalized,
                        backend=backend,
                        fallback_reason=fallback_reason,
                    ),
                    request_id=request_id,
                )
            return self._run_legacy_direct_tts_blocking(
                normalized,
                backend=backend,
                fallback_reason=fallback_reason,
            )
        if normalized.response_streaming:
            return DirectTTSExecution(
                media_type=media_type,
                streaming=True,
                audio_generator=self._iter_legacy_direct_tts_bytes(
                    normalized,
                    backend="legacy_direct_sync_compat",
                    fallback_reason="sync_direct_compat",
                ),
                request_id=request_id,
            )
        return self._run_legacy_direct_tts_blocking(
            normalized,
            backend="legacy_direct_sync_compat",
            fallback_reason="sync_direct_compat",
        )
