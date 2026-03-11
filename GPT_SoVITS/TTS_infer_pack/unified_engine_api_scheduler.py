from __future__ import annotations

import asyncio
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec, T2SFinishedItem, T2SRequestState, run_scheduler_continuous
from GPT_SoVITS.TTS_infer_pack.unified_engine_audio import pack_audio, set_scheduler_seed
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineStatus, NormalizedEngineRequest, SchedulerDebugExecution, SchedulerSubmitExecution


class EngineApiSchedulerFlow:
    def __init__(self, api: Any) -> None:
        self.api = api

    def _build_scheduler_request_specs(self, request_items: List[dict]) -> List[SchedulerRequestSpec]:
        specs: List[SchedulerRequestSpec] = []
        for index, payload in enumerate(request_items):
            normalized = self.api._normalize_engine_request(
                payload,
                request_id=str(payload.get("request_id") or f"req_{index:03d}"),
                error_prefix=f"request[{index}] 参数非法: ",
            )
            specs.append(normalized.to_scheduler_spec())
        return specs

    def _build_scheduler_submit_spec(self, payload: dict | NormalizedEngineRequest) -> SchedulerRequestSpec:
        normalized = self.api._normalize_engine_request(
            payload,
            request_id=(
                payload.request_id
                if isinstance(payload, NormalizedEngineRequest)
                else str(payload.get("request_id") or f"job_{uuid.uuid4().hex[:12]}")
            ),
        )
        return normalized.to_scheduler_spec()

    @staticmethod
    def _summarize_scheduler_states(states: List[T2SRequestState]) -> List[dict]:
        return [
            {
                "request_id": state.request_id,
                "ready_step": int(state.ready_step),
                "ref_audio_path": str(state.ref_audio_path),
                "prompt_semantic_len": int(state.prompt_semantic.shape[0]),
                "all_phone_len": int(state.all_phones.shape[0]),
                "bert_len": int(state.all_bert_features.shape[-1]),
                "norm_text": state.norm_text,
            }
            for state in states
        ]

    @staticmethod
    def _summarize_scheduler_finished(items: List[T2SFinishedItem]) -> List[dict]:
        return [
            {
                "request_id": item.request_id,
                "semantic_len": int(item.semantic_tokens.shape[0]),
                "finish_idx": int(item.finish_idx),
                "finish_reason": item.finish_reason,
            }
            for item in items
        ]

    async def run_scheduler_debug(self, request_items: List[dict], max_steps: int, seed: int) -> SchedulerDebugExecution:
        request_start = time.perf_counter()
        set_scheduler_seed(seed)
        specs = self._build_scheduler_request_specs(request_items)
        request_ids = [spec.request_id for spec in specs]
        for spec in specs:
            self.api._register_request_state(
                request_id=spec.request_id,
                api_mode="scheduler_debug",
                backend="scheduler_debug",
                media_type="wav",
                response_streaming=False,
                meta={
                    "text_len": len(spec.text),
                    "prompt_text_len": len(spec.prompt_text),
                    "text_lang": spec.text_lang,
                    "prompt_lang": spec.prompt_lang,
                    "ref_audio_path": str(spec.ref_audio_path),
                    "ready_step": int(spec.ready_step),
                },
            )
            self.api._update_request_state(spec.request_id, EngineStatus.VALIDATED, {"request_source": "scheduler_debug"})
            self.api._update_request_state(spec.request_id, EngineStatus.CPU_PREPARING, None)
        prepare_started_at = time.perf_counter()
        try:
            states = await self.api.scheduler_worker.prepare_states_batch_async(specs)
        except Exception as exc:
            for request_id in request_ids:
                self.api._fail_request_state(request_id, str(exc))
            raise
        prepare_finished_at = time.perf_counter()
        prepare_batch_wall_ms = max(0.0, (prepare_finished_at - prepare_started_at) * 1000.0)
        for state in states:
            self.api._update_request_state(
                state.request_id,
                EngineStatus.ACTIVE_DECODE,
                {
                    "prepare_profile": dict(state.prepare_profile),
                    "norm_text": state.norm_text,
                    "norm_prompt_text": state.norm_prompt_text,
                },
            )
        decode_started_at = time.perf_counter()
        try:
            finished = run_scheduler_continuous(self.api.tts.t2s_model.model, states, max_steps=int(max_steps))
        except Exception as exc:
            for request_id in request_ids:
                self.api._fail_request_state(request_id, str(exc))
            raise
        decode_finished_at = time.perf_counter()
        decode_batch_wall_ms = max(0.0, (decode_finished_at - decode_started_at) * 1000.0)
        request_total_ms = max(0.0, (decode_finished_at - request_start) * 1000.0)
        finished_map = {item.request_id: item for item in finished}
        request_profiles: List[Dict[str, Any]] = []
        for state in states:
            item = finished_map.get(state.request_id)
            if item is None:
                self.api._fail_request_state(state.request_id, "scheduler_debug finished without result")
                continue
            request_profile = self.api._build_scheduler_debug_request_profile(
                state=state,
                item=item,
                batch_request_count=len(states),
                prepare_batch_wall_ms=prepare_batch_wall_ms,
                decode_batch_wall_ms=decode_batch_wall_ms,
                batch_request_total_ms=request_total_ms,
            )
            request_profiles.append(
                {
                    "request_id": state.request_id,
                    "profile": dict(request_profile),
                }
            )
            self.api._complete_request_state(
                state.request_id,
                dict(request_profile),
            )
        return SchedulerDebugExecution(
            payload={
                "message": "success",
                "request_count": len(states),
                "max_steps": int(max_steps),
                "batch_profile": self.api._build_scheduler_debug_batch_profile(
                    request_count=len(states),
                    max_steps=int(max_steps),
                    prepare_batch_wall_ms=prepare_batch_wall_ms,
                    decode_batch_wall_ms=decode_batch_wall_ms,
                    request_total_ms=request_total_ms,
                    finished_items=finished,
                ),
                "requests": self._summarize_scheduler_states(states),
                "finished": self._summarize_scheduler_finished(finished),
                "request_profiles": request_profiles,
                "request_traces": self.api._collect_request_summaries(request_ids),
            }
        )

    async def run_scheduler_submit(self, payload: dict) -> SchedulerSubmitExecution:
        request_start = time.perf_counter()
        prepare_start = request_start
        normalized = self.api._normalize_engine_request(
            payload,
            request_id=str(payload.get("request_id") or f"job_{uuid.uuid4().hex[:12]}"),
        )
        spec = self._build_scheduler_submit_spec(normalized)
        deadline_ts = None
        timeout_sec = normalized.timeout_sec
        if timeout_sec is not None:
            try:
                deadline_ts = request_start + float(timeout_sec)
            except Exception:
                deadline_ts = None
        self.api._register_request_state(
            request_id=spec.request_id,
            api_mode="scheduler_submit",
            backend="scheduler_v1",
            media_type=normalized.media_type,
            response_streaming=False,
            deadline_ts=deadline_ts,
            meta=self.api._build_request_meta(normalized.to_payload()),
        )
        self.api._update_request_state(spec.request_id, EngineStatus.VALIDATED, {"request_source": "scheduler_submit"})
        spec_ready_at = time.perf_counter()
        prepare_spec_build_ms = max(0.0, (spec_ready_at - prepare_start) * 1000.0)
        self.api._update_request_state(spec.request_id, EngineStatus.CPU_PREPARING, {"prepare_spec_build_ms": prepare_spec_build_ms})
        try:
            state, prepare_exec_started_at, prepare_exec_finished_at = await self.api._prepare_state_via_engine_gpu_queue(
                spec=spec,
                prepare_submit_at=spec_ready_at,
                engine_request_id=spec.request_id,
            )
        except Exception as exc:
            self.api._fail_request_state(spec.request_id, str(exc))
            raise
        prepare_wall_ms = max(0.0, (prepare_exec_finished_at - spec_ready_at) * 1000.0)
        prepare_executor_queue_ms = max(0.0, (prepare_exec_started_at - spec_ready_at) * 1000.0)
        prepare_executor_run_ms = max(0.0, (prepare_exec_finished_at - prepare_exec_started_at) * 1000.0)
        prepare_profile = dict(state.prepare_profile)
        prepare_profile_total_ms = float(prepare_profile.get("wall_total_ms", prepare_wall_ms))
        prepare_profile_wall_ms = float(prepare_profile.get("wall_total_ms", prepare_wall_ms))
        prepare_other_ms = max(0.0, prepare_wall_ms - prepare_spec_build_ms - prepare_executor_queue_ms - prepare_executor_run_ms)
        self.api._update_request_state(
            spec.request_id,
            EngineStatus.READY_FOR_PREFILL,
            {
                "prepare_wall_ms": prepare_wall_ms,
                "prepare_profile_total_ms": prepare_profile_total_ms,
                "prepare_profile": prepare_profile,
            },
        )
        api_after_prepare_start = time.perf_counter()
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        await self.api._enqueue_prepared_state_for_dispatch(
            state=state,
            speed_factor=float(normalized.speed_factor),
            sample_steps=int(normalized.sample_steps),
            media_type=normalized.media_type,
            prepare_wall_ms=prepare_wall_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            done_loop=loop,
            done_future=done_future,
            engine_request_id=spec.request_id,
            timeout_sec=normalized.timeout_sec,
        )
        api_after_prepare_ms = max(0.0, (time.perf_counter() - api_after_prepare_start) * 1000.0)
        try:
            job = await asyncio.wait_for(done_future, timeout=float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0))
        except Exception as exc:
            self.api._fail_request_state(spec.request_id, str(exc))
            raise
        wait_return_at = time.perf_counter()
        if job.error is not None:
            raise RuntimeError(job.error)
        if job.audio_data is None or job.sample_rate is None or job.result is None:
            self.api._fail_request_state(spec.request_id, f"{job.request_id} finished without audio result")
            raise RuntimeError(f"{job.request_id} finished without audio result")
        pack_start = time.perf_counter()
        audio_data = pack_audio(BytesIO(), job.audio_data, int(job.sample_rate), job.media_type).getvalue()
        pack_end = time.perf_counter()
        pack_ms = (pack_end - pack_start) * 1000.0
        api_wait_result_ms = 0.0
        if job.result_ready_time is not None:
            api_wait_result_ms = max(0.0, (wait_return_at - job.result_ready_time) * 1000.0)
        response_ready_at = time.perf_counter()
        response_overhead_ms = max(0.0, (response_ready_at - pack_end) * 1000.0)
        submit_profile = self.api._build_scheduler_submit_profile(
            backend="scheduler_v1",
            request_start=request_start,
            response_ready_at=response_ready_at,
            audio_bytes=len(audio_data),
            sample_rate=int(job.sample_rate),
            prepare_spec_build_ms=prepare_spec_build_ms,
            prepare_wall_ms=prepare_wall_ms,
            prepare_executor_queue_ms=prepare_executor_queue_ms,
            prepare_executor_run_ms=prepare_executor_run_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            prepare_profile_wall_ms=prepare_profile_wall_ms,
            prepare_other_ms=prepare_other_ms,
            engine_policy_wait_ms=float(job.result.get("engine_policy_wait_ms", 0.0)),
            api_after_prepare_ms=api_after_prepare_ms,
            api_wait_result_ms=api_wait_result_ms,
            pack_ms=pack_ms,
            response_overhead_ms=response_overhead_ms,
            worker_profile=dict(job.result or {}),
        )
        headers = self.api._build_scheduler_submit_headers(
            request_id=job.request_id,
            media_type=job.media_type,
            sample_rate=int(job.sample_rate),
            profile=submit_profile,
        )
        self.api._merge_request_state_profile(
            spec.request_id,
            dict(submit_profile, response_headers_emitted=True),
        )
        return SchedulerSubmitExecution(audio_bytes=audio_data, media_type=f"audio/{job.media_type}", headers=headers)
