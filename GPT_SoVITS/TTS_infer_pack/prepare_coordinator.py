from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import (
    PreparedTextFeatures,
    SchedulerRequestSpec,
    T2SRequestState,
    build_empty_text_features,
    build_request_state_from_parts,
    normalize_sentence,
)


@dataclass
class ProfiledResult:
    result: Any
    submit_at: float
    started_at: float
    finished_at: float
    profile: Dict[str, float] | None = None

    @property
    def queue_ms(self) -> float:
        return max(0.0, (self.started_at - self.submit_at) * 1000.0)

    @property
    def run_ms(self) -> float:
        return max(0.0, (self.finished_at - self.started_at) * 1000.0)


@dataclass
class PreparedCpuStage:
    spec: SchedulerRequestSpec
    prepare_submit_at: float
    prepare_start: float
    prompt_text: str
    text: str
    prepare_admission_wait_ms: float
    current_inflight: int
    peak_inflight: int
    prompt_cpu_profiled: ProfiledResult
    target_cpu_profiled: ProfiledResult


class AsyncStageGate:
    def __init__(self, max_inflight: int, poll_ms: int = 1):
        self.max_inflight = max(0, int(max_inflight))
        self.lock = threading.Lock()
        self.poll_s = max(0.0005, float(max(1, int(poll_ms))) / 1000.0)
        self.inflight = 0
        self.peak_inflight = 0
        self.total_entered = 0
        self.total_wait_ms = 0.0
        self.wait_peak_ms = 0.0

    async def acquire(self) -> Dict[str, float]:
        wait_start = time.perf_counter()
        while True:
            with self.lock:
                if self.max_inflight <= 0 or self.inflight < self.max_inflight:
                    self.inflight += 1
                    self.total_entered += 1
                    wait_ms = max(0.0, (time.perf_counter() - wait_start) * 1000.0)
                    self.total_wait_ms += float(wait_ms)
                    self.wait_peak_ms = max(self.wait_peak_ms, float(wait_ms))
                    self.peak_inflight = max(self.peak_inflight, self.inflight)
                    return {
                        "wait_ms": float(wait_ms),
                        "inflight": float(self.inflight),
                        "peak_inflight": float(self.peak_inflight),
                        "max_inflight": float(self.max_inflight),
                    }
            await asyncio.sleep(self.poll_s)

    def release(self) -> None:
        with self.lock:
            self.inflight = max(0, self.inflight - 1)

    def snapshot(self) -> Dict[str, float]:
        with self.lock:
            return {
                "max_inflight": float(self.max_inflight),
                "inflight": float(self.inflight),
                "peak_inflight": float(self.peak_inflight),
                "total_entered": float(self.total_entered),
                "total_wait_ms": float(self.total_wait_ms),
                "wait_peak_ms": float(self.wait_peak_ms),
            }


class PrepareCoordinator:
    def __init__(self, tts: Any):
        self.tts = tts
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak_inflight = 0
        self.use_async_text_feature_path = bool(
            getattr(tts, "prepare_bert_batch_worker", None) is not None
            and os.environ.get("GPTSOVITS_PREPARE_TEXT_FEATURE_DIRECT", "0") != "0"
        )
        self.max_inflight = max(0, int(os.environ.get("GPTSOVITS_PREPARE_MAX_INFLIGHT", "0")))
        gate_poll_ms = int(os.environ.get("GPTSOVITS_PREPARE_GATE_POLL_MS", "1"))
        self._inflight_gate = AsyncStageGate(self.max_inflight, poll_ms=gate_poll_ms)
        self.text_feature_workers = 0
        self.text_feature_executor = None
        if not self.use_async_text_feature_path:
            text_feature_default_workers = max(1, int(getattr(tts, "prepare_text_cpu_workers", 16) or 16))
            self.text_feature_workers = max(
                1,
                int(os.environ.get("GPTSOVITS_PREPARE_TEXT_FEATURE_WORKERS", str(text_feature_default_workers))),
            )
            self.text_feature_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.text_feature_workers,
                thread_name_prefix="prepare-text-feature",
            )
        g2pw_default_workers = max(8, int(getattr(tts, "prepare_text_cpu_workers", 8) or 8))
        self.g2pw_workers = max(
            1,
            int(os.environ.get("GPTSOVITS_PREPARE_G2PW_WORKERS", str(g2pw_default_workers))),
        )
        self.g2pw_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.g2pw_workers,
            thread_name_prefix="prepare-g2pw",
        )
        ref_audio_default_workers = max(1, int(os.environ.get("GPTSOVITS_PREPARE_REF_SLOTS", "4")))
        self.ref_audio_workers = max(
            1,
            int(os.environ.get("GPTSOVITS_PREPARE_REF_ASYNC_WORKERS", str(ref_audio_default_workers))),
        )
        self.ref_audio_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.ref_audio_workers,
            thread_name_prefix="prepare-ref-audio",
        )
        text_cpu_gate_default = max(0, int(getattr(tts, "prepare_text_cpu_workers", 0) or 0))
        g2pw_gate_default = max(0, int(self.g2pw_workers))
        text_feature_gate_default = max(0, int(self.text_feature_workers))
        ref_audio_gate_default = max(0, int(self.ref_audio_workers))
        self.text_cpu_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_MAX_INFLIGHT", str(text_cpu_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.g2pw_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_G2PW_MAX_INFLIGHT", str(g2pw_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.text_feature_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_TEXT_FEATURE_MAX_INFLIGHT", str(text_feature_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.ref_audio_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_REF_MAX_INFLIGHT", str(ref_audio_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.ref_load_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_REF_LOAD_MAX_INFLIGHT", str(ref_audio_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.ref_spec_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_REF_SPEC_MAX_INFLIGHT", str(ref_audio_gate_default))),
            poll_ms=gate_poll_ms,
        )

    def _mark_enter(self) -> Tuple[int, int]:
        with self.lock:
            self.inflight += 1
            current_inflight = self.inflight
            if current_inflight > self.peak_inflight:
                self.peak_inflight = current_inflight
            return current_inflight, self.peak_inflight

    def _mark_leave(self) -> None:
        with self.lock:
            self.inflight = max(0, self.inflight - 1)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            snapshot: Dict[str, Any] = {
                "inflight": int(self.inflight),
                "peak_inflight": int(self.peak_inflight),
                "max_inflight": int(self.max_inflight),
                "text_feature_workers": int(self.text_feature_workers),
                "g2pw_workers": int(self.g2pw_workers),
                "ref_audio_workers": int(self.ref_audio_workers),
            }
        runtime_snapshot_fn = getattr(self.tts, "snapshot_prepare_runtime_components", None)
        if callable(runtime_snapshot_fn):
            try:
                snapshot["prepare_runtime_state"] = runtime_snapshot_fn()
            except Exception:
                snapshot["prepare_runtime_state"] = None
        snapshot["prepare_stage_gates"] = {
            "text_cpu": self.text_cpu_gate.snapshot(),
            "g2pw": self.g2pw_gate.snapshot(),
            "text_feature": self.text_feature_gate.snapshot(),
            "ref_audio": self.ref_audio_gate.snapshot(),
            "ref_load": self.ref_load_gate.snapshot(),
            "ref_spec": self.ref_spec_gate.snapshot(),
        }
        return snapshot

    @staticmethod
    def _run_profiled(fn, submit_at: float, *args) -> ProfiledResult:
        started_at = time.perf_counter()
        result = fn(*args)
        finished_at = time.perf_counter()
        return ProfiledResult(
            result=result,
            submit_at=float(submit_at),
            started_at=float(started_at),
            finished_at=float(finished_at),
        )

    def _prepare_text_cpu(self, text: str, language: str):
        return self.tts.prepare_text_segments(text, language)

    def _resolve_g2pw_segments(self, prepared_segments):
        profile: Dict[str, float] = {}
        resolved_segments = self.tts.resolve_g2pw_segments(prepared_segments, profile=profile)
        return resolved_segments, profile

    def _load_ref_audio_raw(self, ref_audio_path: str):
        return self.tts._load_ref_audio_raw(ref_audio_path)

    def _build_ref_prompt_semantic_from_raw(self, raw_audio, raw_sr: int):
        load_profile = {"audio_load_ms": 0.0}
        if getattr(self.tts, "prepare_ref_semantic_batch_worker", None) is not None:
            prompt_semantic, worker_profile = self.tts.prepare_ref_semantic_batch_worker.submit(raw_audio, raw_sr)
            return {
                "prompt_semantic": prompt_semantic,
                "raw_audio": raw_audio,
                "raw_sr": raw_sr,
                "profile": {
                    **load_profile,
                    "audio_stage_wait_ms": float(worker_profile.get("prompt_semantic_wait_ms", 0.0)),
                    "audio_stage_slots": float(worker_profile.get("prompt_semantic_stage_slots", 0.0)),
                    "audio_stage_inflight_peak": float(worker_profile.get("prompt_semantic_stage_inflight_peak", 0.0)),
                    "prompt_semantic_ms": float(
                        worker_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)
                        + worker_profile.get("prompt_semantic_forward_ms", 0.0)
                        + worker_profile.get("prompt_semantic_scatter_ms", 0.0)
                    ),
                    **{key: float(value) for key, value in worker_profile.items()},
                    "ref_spec_wait_ms": 0.0,
                    "ref_spec_ms": 0.0,
                    "bundle_total_ms": float(worker_profile.get("prompt_semantic_wait_ms", 0.0))
                    + float(worker_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
                    + float(worker_profile.get("prompt_semantic_forward_ms", 0.0))
                    + float(worker_profile.get("prompt_semantic_scatter_ms", 0.0)),
                },
            }
        wav16k, cpu_prepare_ms, limiter_stats = self.tts._prepare_prompt_semantic_wav16k_profile(raw_audio, raw_sr)
        with self.tts.prepare_ref_audio_stage_limiter.enter() as stage_stats:
            prompt_semantic, forward_ms = self.tts._extract_prompt_semantic_profile_from_prepared_wav16k(wav16k)
        return {
            "prompt_semantic": prompt_semantic,
            "raw_audio": raw_audio,
            "raw_sr": raw_sr,
            "profile": {
                "audio_load_ms": 0.0,
                "audio_stage_wait_ms": float(stage_stats.get("wait_ms", 0.0)),
                "audio_stage_slots": float(stage_stats.get("slots", 0.0)),
                "audio_stage_inflight_peak": float(stage_stats.get("peak_inflight", 0.0)),
                "prompt_semantic_wait_ms": float(stage_stats.get("wait_ms", 0.0)),
                "prompt_semantic_cpu_prepare_wait_ms": float(limiter_stats.get("wait_ms", 0.0)),
                "prompt_semantic_cpu_prepare_slots": float(limiter_stats.get("slots", 0.0)),
                "prompt_semantic_cpu_prepare_inflight_peak": float(limiter_stats.get("peak_inflight", 0.0)),
                "prompt_semantic_worker_queue_wait_ms": 0.0,
                "prompt_semantic_batch_collect_wait_ms": 0.0,
                "prompt_semantic_stage_limiter_wait_ms": float(stage_stats.get("wait_ms", 0.0)),
                "prompt_semantic_batch_dispatch_delay_ms": 0.0,
                "prompt_semantic_cpu_prepare_ms": float(cpu_prepare_ms),
                "prompt_semantic_pack_ms": 0.0,
                "prompt_semantic_h2d_ms": 0.0,
                "prompt_semantic_ssl_forward_ms": 0.0,
                "prompt_semantic_hidden_length_ms": 0.0,
                "prompt_semantic_extract_latent_ms": 0.0,
                "prompt_semantic_forward_ms": float(forward_ms),
                "prompt_semantic_scatter_ms": 0.0,
                "prompt_semantic_stage_slots": float(stage_stats.get("slots", 0.0)),
                "prompt_semantic_stage_inflight_peak": float(stage_stats.get("peak_inflight", 0.0)),
                "prompt_semantic_batch_size": 1.0,
                "prompt_semantic_batch_samples": 0.0,
                "ref_spec_wait_ms": 0.0,
                "ref_spec_ms": 0.0,
                "bundle_total_ms": float(cpu_prepare_ms + forward_ms + stage_stats.get("wait_ms", 0.0)),
            },
        }

    def _extract_ref_spec_from_raw(self, raw_audio, raw_sr: int):
        spec, audio, _, _, profile = self.tts._extract_ref_spec_profile_from_raw(raw_audio, raw_sr)
        return (spec, audio), profile

    @staticmethod
    def _build_empty_text_features_like(reference: PreparedTextFeatures | None = None) -> PreparedTextFeatures:
        feature_dim = 1024
        dtype = None
        if reference is not None:
            try:
                feature_dim = int(reference.bert_features.shape[0])
                dtype = reference.bert_features.dtype
            except Exception:
                pass
        return build_empty_text_features(
            feature_dim=int(feature_dim),
            dtype=(dtype if dtype is not None else None) or __import__("torch").float32,
        )

    def _build_text_features(
        self,
        prepared_segments,
        language: str,
        cpu_run_ms: float,
        base_profile: Dict[str, float] | None = None,
    ) -> PreparedTextFeatures:
        profile: Dict[str, float] = dict(base_profile or {})
        profile["cpu_preprocess_ms"] = float(cpu_run_ms)
        branch_start = time.perf_counter()
        phones, bert_features, norm_text = self.tts.build_text_features_from_segments(prepared_segments, profile=profile)
        total_ms = float(cpu_run_ms + (time.perf_counter() - branch_start) * 1000.0)
        profile["bert_total_ms"] = max(0.0, total_ms - float(cpu_run_ms))
        return PreparedTextFeatures(
            phones=phones,
            bert_features=bert_features,
            norm_text=norm_text,
            profile=profile,
            total_ms=total_ms,
            cpu_preprocess_ms=float(cpu_run_ms),
        )

    async def _run_on_executor(self, executor, fn, *args) -> ProfiledResult:
        loop = asyncio.get_running_loop()
        submit_at = time.perf_counter()
        return await loop.run_in_executor(executor, self._run_profiled, fn, float(submit_at), *args)

    async def _run_text_cpu_stage(self, text: str, language: str) -> ProfiledResult:
        await self.text_cpu_gate.acquire()
        if text in [None, ""]:
            try:
                submit_at = time.perf_counter()
                return ProfiledResult(result=[], submit_at=submit_at, started_at=submit_at, finished_at=submit_at)
            finally:
                self.text_cpu_gate.release()
        text_cpu_worker = getattr(self.tts, "prepare_text_cpu_worker", None)
        executor = getattr(self.tts, "prepare_text_cpu_executor", None)
        try:
            if text_cpu_worker is not None:
                submit_at = time.perf_counter()
                result, worker_profile = await text_cpu_worker.submit_async(text, language)
                started_at = float(
                    submit_at
                    + (
                        float(worker_profile.get("text_cpu_admission_wait_ms", 0.0))
                        + float(worker_profile.get("text_cpu_queue_wait_ms", 0.0))
                    )
                    / 1000.0
                )
                finished_at = float(started_at + float(worker_profile.get("text_cpu_run_ms", 0.0)) / 1000.0)
                return ProfiledResult(
                    result=result,
                    submit_at=float(submit_at),
                    started_at=started_at,
                    finished_at=finished_at,
                    profile=dict(worker_profile),
                )
            if executor is None:
                submit_at = time.perf_counter()
                return self._run_profiled(self._prepare_text_cpu, submit_at, text, language)
            return await self._run_on_executor(executor, self._prepare_text_cpu, text, language)
        finally:
            self.text_cpu_gate.release()

    async def _run_text_feature_stage(self, prepared_segments, language: str, cpu_run_ms: float) -> ProfiledResult:
        await self.text_feature_gate.acquire()
        try:
            return await self._run_on_executor(
                self.text_feature_executor,
                self._build_text_features,
                prepared_segments,
                language,
                cpu_run_ms,
                None,
            )
        finally:
            self.text_feature_gate.release()

    async def _run_g2pw_stage(self, prepared_segments) -> ProfiledResult:
        has_pending = any(bool(getattr(segment, "needs_g2pw", False)) for segment in (prepared_segments or []))
        if not has_pending:
            submit_at = time.perf_counter()
            return ProfiledResult(
                result=prepared_segments,
                submit_at=float(submit_at),
                started_at=float(submit_at),
                finished_at=float(submit_at),
                profile={},
            )
        await self.g2pw_gate.acquire()
        try:
            profiled = await self._run_on_executor(self.g2pw_executor, self._resolve_g2pw_segments, prepared_segments)
            result, stage_profile = profiled.result
            return ProfiledResult(
                result=result,
                submit_at=float(profiled.submit_at),
                started_at=float(profiled.started_at),
                finished_at=float(profiled.finished_at),
                profile=dict(stage_profile),
            )
        finally:
            self.g2pw_gate.release()

    async def _run_g2pw_pair_stage(self, prompt_segments, target_segments) -> tuple[ProfiledResult, ProfiledResult]:
        prompt_is_empty = len(prompt_segments or []) == 0
        target_task = asyncio.create_task(self._run_g2pw_stage(target_segments))
        if not prompt_is_empty:
            prompt_task = asyncio.create_task(self._run_g2pw_stage(prompt_segments))
            return await asyncio.gather(prompt_task, target_task)
        target_profiled = await target_task
        submit_at = time.perf_counter()
        prompt_profiled = ProfiledResult(
            result=prompt_segments,
            submit_at=float(submit_at),
            started_at=float(submit_at),
            finished_at=float(submit_at),
            profile={},
        )
        return prompt_profiled, target_profiled

    @staticmethod
    def _estimate_text_feature_run_ms(profile: Dict[str, float]) -> float:
        return float(
            profile.get("bert_wait_ms", 0.0)
            + profile.get("bert_tokenize_ms", 0.0)
            + profile.get("bert_forward_ms", 0.0)
            + profile.get("bert_scatter_ms", 0.0)
        )

    async def _run_text_feature_pair_stage(
        self,
        prompt_segments,
        target_segments,
        prompt_cpu_run_ms: float,
        target_cpu_run_ms: float,
        prompt_base_profile: Dict[str, float] | None = None,
        target_base_profile: Dict[str, float] | None = None,
    ) -> tuple[ProfiledResult, ProfiledResult]:
        prompt_is_empty = len(prompt_segments or []) == 0
        if self.text_feature_executor is not None:
            target_feature_task = asyncio.create_task(
                self._run_on_executor(
                    self.text_feature_executor,
                    self._build_text_features,
                    target_segments,
                    None,
                    target_cpu_run_ms,
                    target_base_profile,
                )
            )
            if not prompt_is_empty:
                prompt_feature_task = asyncio.create_task(
                    self._run_on_executor(
                        self.text_feature_executor,
                        self._build_text_features,
                        prompt_segments,
                        None,
                        prompt_cpu_run_ms,
                        prompt_base_profile,
                    )
                )
                return await asyncio.gather(prompt_feature_task, target_feature_task)
            target_profiled = await target_feature_task
            submit_at = time.perf_counter()
            prompt_profiled = ProfiledResult(
                result=self._build_empty_text_features_like(target_profiled.result),
                submit_at=float(submit_at),
                started_at=float(submit_at),
                finished_at=float(submit_at),
            )
            return prompt_profiled, target_profiled

        await self.text_feature_gate.acquire()
        target_profile: Dict[str, float] = dict(target_base_profile or {})
        target_profile["cpu_preprocess_ms"] = float(target_cpu_run_ms)
        submit_at = time.perf_counter()
        started_at = float(submit_at)
        try:
            if prompt_is_empty:
                target_result_raw = await self.tts.build_text_features_from_segments_async(
                    target_segments,
                    profile=target_profile,
                )
                prompt_result = self._build_empty_text_features_like(
                    PreparedTextFeatures(
                        phones=target_result_raw[0],
                        bert_features=target_result_raw[1],
                        norm_text=target_result_raw[2],
                        profile=target_profile,
                        total_ms=float(target_cpu_run_ms + self._estimate_text_feature_run_ms(target_profile)),
                        cpu_preprocess_ms=float(target_cpu_run_ms),
                    )
                )
                finished_at = time.perf_counter()
                prompt_profiled = ProfiledResult(
                    result=prompt_result,
                    submit_at=float(submit_at),
                    started_at=float(submit_at),
                    finished_at=float(submit_at),
                )
                target_result = PreparedTextFeatures(
                    phones=target_result_raw[0],
                    bert_features=target_result_raw[1],
                    norm_text=target_result_raw[2],
                    profile=target_profile,
                    total_ms=float(target_cpu_run_ms + self._estimate_text_feature_run_ms(target_profile)),
                    cpu_preprocess_ms=float(target_cpu_run_ms),
                )
                target_profiled = ProfiledResult(
                    result=target_result,
                    submit_at=float(submit_at),
                    started_at=started_at,
                    finished_at=float(submit_at + self._estimate_text_feature_run_ms(target_profile) / 1000.0),
                )
                if finished_at > target_profiled.finished_at:
                    target_result.profile["bert_total_ms"] = max(
                        self._estimate_text_feature_run_ms(target_profile),
                        (finished_at - submit_at) * 1000.0,
                    )
                else:
                    target_result.profile["bert_total_ms"] = self._estimate_text_feature_run_ms(target_profile)
                return prompt_profiled, target_profiled

            prompt_profile: Dict[str, float] = dict(prompt_base_profile or {})
            prompt_profile["cpu_preprocess_ms"] = float(prompt_cpu_run_ms)
            prompt_result_raw, target_result_raw = await self.tts.build_text_feature_pair_from_segments_async(
                prompt_segments,
                target_segments,
                prompt_profile=prompt_profile,
                target_profile=target_profile,
            )
            finished_at = time.perf_counter()

            prompt_result = PreparedTextFeatures(
                phones=prompt_result_raw[0],
                bert_features=prompt_result_raw[1],
                norm_text=prompt_result_raw[2],
                profile=prompt_profile,
                total_ms=float(prompt_cpu_run_ms + self._estimate_text_feature_run_ms(prompt_profile)),
                cpu_preprocess_ms=float(prompt_cpu_run_ms),
            )
            target_result = PreparedTextFeatures(
                phones=target_result_raw[0],
                bert_features=target_result_raw[1],
                norm_text=target_result_raw[2],
                profile=target_profile,
                total_ms=float(target_cpu_run_ms + self._estimate_text_feature_run_ms(target_profile)),
                cpu_preprocess_ms=float(target_cpu_run_ms),
            )
            prompt_profiled = ProfiledResult(
                result=prompt_result,
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(submit_at + self._estimate_text_feature_run_ms(prompt_profile) / 1000.0),
            )
            target_profiled = ProfiledResult(
                result=target_result,
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(submit_at + self._estimate_text_feature_run_ms(target_profile) / 1000.0),
            )
            if finished_at > prompt_profiled.finished_at:
                prompt_result.profile["bert_total_ms"] = max(
                    self._estimate_text_feature_run_ms(prompt_profile),
                    (finished_at - submit_at) * 1000.0,
                )
                target_result.profile["bert_total_ms"] = max(
                    self._estimate_text_feature_run_ms(target_profile),
                    (finished_at - submit_at) * 1000.0,
                )
            else:
                prompt_result.profile["bert_total_ms"] = self._estimate_text_feature_run_ms(prompt_profile)
                target_result.profile["bert_total_ms"] = self._estimate_text_feature_run_ms(target_profile)
            return prompt_profiled, target_profiled
        finally:
            self.text_feature_gate.release()

    async def _run_ref_prompt_semantic_stage(self, ref_audio_path: str) -> ProfiledResult:
        if getattr(self.tts, "prepare_ref_semantic_batch_worker", None) is not None:
            submit_at = time.perf_counter()
            started_at = float(submit_at)

            await self.ref_load_gate.acquire()
            try:
                load_profiled = await self._run_on_executor(self.ref_audio_executor, self._load_ref_audio_raw, ref_audio_path)
            finally:
                self.ref_load_gate.release()

            raw_audio, raw_sr = load_profiled.result
            prompt_semantic_task = asyncio.create_task(
                self.tts.prepare_ref_semantic_batch_worker.submit_async(raw_audio, raw_sr)
            )
            prompt_semantic, prompt_semantic_profile = await prompt_semantic_task
            limiter_snapshot = (
                self.tts.prepare_ref_audio_stage_limiter.snapshot()
                if getattr(self.tts, "prepare_ref_audio_stage_limiter", None) is not None
                else {}
            )
            prompt_semantic_ms = (
                float(prompt_semantic_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
                + float(prompt_semantic_profile.get("prompt_semantic_forward_ms", 0.0))
                + float(prompt_semantic_profile.get("prompt_semantic_scatter_ms", 0.0))
            )
            finished_at = time.perf_counter()
            result = {
                "prompt_semantic": prompt_semantic,
                "raw_audio": raw_audio,
                "raw_sr": raw_sr,
                "profile": {
                    "audio_load_queue_ms": float(load_profiled.queue_ms),
                    "audio_load_ms": float(load_profiled.run_ms),
                    "audio_stage_wait_ms": float(prompt_semantic_profile.get("prompt_semantic_wait_ms", 0.0)),
                    "audio_stage_slots": float(
                        max(
                            float(prompt_semantic_profile.get("prompt_semantic_stage_slots", 0.0)),
                            float(limiter_snapshot.get("slots", 0.0)),
                        )
                    ),
                    "audio_stage_inflight_peak": float(
                        max(
                            float(prompt_semantic_profile.get("prompt_semantic_stage_inflight_peak", 0.0)),
                            float(limiter_snapshot.get("peak_inflight", 0.0)),
                        )
                    ),
                    "prompt_semantic_ms": float(prompt_semantic_ms),
                    "prompt_semantic_wait_ms": float(prompt_semantic_profile.get("prompt_semantic_wait_ms", 0.0)),
                    "prompt_semantic_cpu_prepare_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)
                    ),
                    "prompt_semantic_pack_ms": float(prompt_semantic_profile.get("prompt_semantic_pack_ms", 0.0)),
                    "prompt_semantic_h2d_ms": float(prompt_semantic_profile.get("prompt_semantic_h2d_ms", 0.0)),
                    "prompt_semantic_ssl_forward_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_ssl_forward_ms", 0.0)
                    ),
                    "prompt_semantic_hidden_length_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_hidden_length_ms", 0.0)
                    ),
                    "prompt_semantic_extract_latent_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_extract_latent_ms", 0.0)
                    ),
                    "prompt_semantic_forward_ms": float(prompt_semantic_profile.get("prompt_semantic_forward_ms", 0.0)),
                    "prompt_semantic_scatter_ms": float(prompt_semantic_profile.get("prompt_semantic_scatter_ms", 0.0)),
                    "prompt_semantic_stage_slots": float(prompt_semantic_profile.get("prompt_semantic_stage_slots", 0.0)),
                    "prompt_semantic_stage_inflight_peak": float(
                        prompt_semantic_profile.get("prompt_semantic_stage_inflight_peak", 0.0)
                    ),
                    "prompt_semantic_batch_size": float(prompt_semantic_profile.get("prompt_semantic_batch_size", 1.0)),
                    "prompt_semantic_batch_samples": float(
                        prompt_semantic_profile.get("prompt_semantic_batch_samples", 0.0)
                    ),
                    "bundle_total_ms": float(
                        load_profiled.queue_ms
                        + load_profiled.run_ms
                        + prompt_semantic_ms
                    ),
                },
            }
            return ProfiledResult(
                result=result,
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(finished_at),
            )

        await self.ref_audio_gate.acquire()
        try:
            load_profiled = await self._run_on_executor(self.ref_audio_executor, self._load_ref_audio_raw, ref_audio_path)
            raw_audio, raw_sr = load_profiled.result
            submit_at = time.perf_counter()
            started_at = time.perf_counter()
            result = await asyncio.to_thread(self._build_ref_prompt_semantic_from_raw, raw_audio, raw_sr)
            result.setdefault("profile", {})
            result["profile"]["audio_load_queue_ms"] = float(load_profiled.queue_ms)
            result["profile"]["audio_load_ms"] = float(load_profiled.run_ms)
            finished_at = time.perf_counter()
            return ProfiledResult(result=result, submit_at=float(submit_at), started_at=float(started_at), finished_at=float(finished_at))
        finally:
            self.ref_audio_gate.release()

    async def _run_ref_spec_stage(self, raw_audio, raw_sr: int) -> ProfiledResult:
        await self.ref_spec_gate.acquire()
        try:
            return await self._run_on_executor(self.ref_audio_executor, self._extract_ref_spec_from_raw, raw_audio, raw_sr)
        finally:
            self.ref_spec_gate.release()

    def _release_split_stage_slot(self) -> None:
        self._mark_leave()
        self._inflight_gate.release()

    async def prepare_cpu_stage_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> PreparedCpuStage:
        admission_start = time.perf_counter()
        admission_stats = await self._inflight_gate.acquire()
        prepare_admission_wait_ms = max(
            float(admission_stats.get("wait_ms", 0.0)),
            (time.perf_counter() - admission_start) * 1000.0,
        )
        current_inflight, peak_inflight = self._mark_enter()
        prepare_start = time.perf_counter()
        prompt_text = normalize_sentence(spec.prompt_text, spec.prompt_lang)
        text = spec.text.strip("\n")
        try:
            prompt_cpu_task = asyncio.create_task(self._run_text_cpu_stage(prompt_text, spec.prompt_lang))
            target_cpu_task = asyncio.create_task(self._run_text_cpu_stage(text, spec.text_lang))
            prompt_cpu_profiled, target_cpu_profiled = await asyncio.gather(prompt_cpu_task, target_cpu_task)
            return PreparedCpuStage(
                spec=spec,
                prepare_submit_at=float(prepare_submit_at),
                prepare_start=float(prepare_start),
                prompt_text=prompt_text,
                text=text,
                prepare_admission_wait_ms=float(prepare_admission_wait_ms),
                current_inflight=int(current_inflight),
                peak_inflight=int(peak_inflight),
                prompt_cpu_profiled=prompt_cpu_profiled,
                target_cpu_profiled=target_cpu_profiled,
            )
        except Exception:
            self._release_split_stage_slot()
            raise

    async def prepare_gpu_stage_profiled_async(
        self,
        cpu_stage: PreparedCpuStage,
    ) -> tuple[T2SRequestState, float, float]:
        try:
            phase_one = await self._prepare_gpu_phase_one(cpu_stage)
            phase_two = await self._prepare_gpu_phase_two(cpu_stage, phase_one)
            return self._build_gpu_prepare_result(
                cpu_stage,
                phase_one,
                phase_two,
                extra_profile={
                    "engine_prepare_audio_phase_mode": 0.0,
                    "engine_prepare_audio_phase_wall_ms": float(phase_one["phase_wall_ms"]),
                    "engine_prepare_audio_phase_batch_size": 1.0,
                    "engine_prepare_text_phase_wall_ms": float(phase_two["phase_wall_ms"]),
                    "engine_prepare_text_phase_batch_size": 1.0,
                },
            )
        finally:
            self._release_split_stage_slot()

    async def _prepare_gpu_phase_one(self, cpu_stage: PreparedCpuStage) -> Dict[str, Any]:
        phase_start = time.perf_counter()
        g2pw_pair_task = asyncio.create_task(
            self._run_g2pw_pair_stage(
                cpu_stage.prompt_cpu_profiled.result,
                cpu_stage.target_cpu_profiled.result,
            )
        )
        ref_audio_task = asyncio.create_task(self._run_ref_prompt_semantic_stage(str(cpu_stage.spec.ref_audio_path)))
        prompt_g2pw_profiled, target_g2pw_profiled = await g2pw_pair_task
        g2pw_pair_end = time.perf_counter()
        ref_audio_profiled = await ref_audio_task
        phase_end = time.perf_counter()
        return {
            "prompt_g2pw_profiled": prompt_g2pw_profiled,
            "target_g2pw_profiled": target_g2pw_profiled,
            "ref_audio_profiled": ref_audio_profiled,
            "ref_spec_result": None,
            "g2pw_pair_ms": max(0.0, (g2pw_pair_end - phase_start) * 1000.0),
            "phase_wall_ms": max(0.0, (phase_end - phase_start) * 1000.0),
        }

    async def _prepare_gpu_phase_two(
        self,
        cpu_stage: PreparedCpuStage,
        phase_one: Dict[str, Any],
    ) -> Dict[str, Any]:
        phase_start = time.perf_counter()
        prompt_g2pw_profiled = phase_one["prompt_g2pw_profiled"]
        target_g2pw_profiled = phase_one["target_g2pw_profiled"]
        prompt_feature_profiled, target_feature_profiled = await self._run_text_feature_pair_stage(
            prompt_g2pw_profiled.result,
            target_g2pw_profiled.result,
            cpu_stage.prompt_cpu_profiled.run_ms,
            cpu_stage.target_cpu_profiled.run_ms,
            prompt_base_profile=dict(prompt_g2pw_profiled.profile or {}),
            target_base_profile=dict(target_g2pw_profiled.profile or {}),
        )
        phase_end = time.perf_counter()
        return {
            "prompt_feature_profiled": prompt_feature_profiled,
            "target_feature_profiled": target_feature_profiled,
            "phase_wall_ms": max(0.0, (phase_end - phase_start) * 1000.0),
        }

    def _build_gpu_prepare_result(
        self,
        cpu_stage: PreparedCpuStage,
        phase_one: Dict[str, Any],
        phase_two: Dict[str, Any],
        extra_profile: Dict[str, float] | None = None,
    ) -> tuple[T2SRequestState, float, float]:
        prompt_g2pw_profiled = phase_one["prompt_g2pw_profiled"]
        target_g2pw_profiled = phase_one["target_g2pw_profiled"]
        ref_audio_profiled = phase_one["ref_audio_profiled"]
        ref_spec_result = phase_one.get("ref_spec_result")
        prompt_feature_profiled = phase_two["prompt_feature_profiled"]
        target_feature_profiled = phase_two["target_feature_profiled"]
        profile_overrides = {
            "executor_queue_ms": max(0.0, (cpu_stage.prepare_start - cpu_stage.prepare_submit_at) * 1000.0),
            "prepare_admission_wait_ms": cpu_stage.prepare_admission_wait_ms,
            "prepare_submit_ts": float(cpu_stage.prepare_submit_at),
            "prepare_cpu_start_ts": float(cpu_stage.prepare_start),
            "prepare_cpu_done_ts": float(
                max(cpu_stage.prompt_cpu_profiled.finished_at, cpu_stage.target_cpu_profiled.finished_at)
            ),
            "prompt_text_cpu_start_ts": float(cpu_stage.prompt_cpu_profiled.started_at),
            "prompt_text_cpu_end_ts": float(cpu_stage.prompt_cpu_profiled.finished_at),
            "text_cpu_start_ts": float(cpu_stage.target_cpu_profiled.started_at),
            "text_cpu_end_ts": float(cpu_stage.target_cpu_profiled.finished_at),
            "executor_run_wall_ms": max(0.0, (time.perf_counter() - cpu_stage.prepare_start) * 1000.0),
            "text_feature_pair_ms": float(phase_two["phase_wall_ms"]),
            "g2pw_pair_ms": float(phase_one["g2pw_pair_ms"]),
            "prompt_text_g2pw_queue_ms": prompt_g2pw_profiled.queue_ms,
            "prompt_text_g2pw_run_ms": prompt_g2pw_profiled.run_ms,
            "prompt_text_g2pw_prepare_ms": float((prompt_g2pw_profiled.profile or {}).get("g2pw_prepare_ms", 0.0)),
            "prompt_text_g2pw_predict_ms": float((prompt_g2pw_profiled.profile or {}).get("g2pw_predict_ms", 0.0)),
            "prompt_text_g2pw_post_ms": float((prompt_g2pw_profiled.profile or {}).get("g2pw_post_ms", 0.0)),
            "text_g2pw_queue_ms": target_g2pw_profiled.queue_ms,
            "text_g2pw_run_ms": target_g2pw_profiled.run_ms,
            "text_g2pw_prepare_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_prepare_ms", 0.0)),
            "text_g2pw_predict_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_predict_ms", 0.0)),
            "text_g2pw_post_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_post_ms", 0.0)),
            "prompt_text_parallel_future_wait_ms": 0.0,
            "prompt_text_parallel_future_executor_queue_ms": 0.0,
            "prompt_text_parallel_future_run_ms": 0.0,
            "prompt_text_parallel_future_finish_after_submit_ms": 0.0,
            "prompt_text_parallel_future_queue_tail_after_target_ms": 0.0,
            "prompt_text_parallel_future_run_tail_after_target_ms": 0.0,
            "prompt_text_cpu_queue_ms": cpu_stage.prompt_cpu_profiled.queue_ms,
            "prompt_text_cpu_run_ms": cpu_stage.prompt_cpu_profiled.run_ms,
            "prompt_text_cpu_admission_wait_ms": float(
                (cpu_stage.prompt_cpu_profiled.profile or {}).get("text_cpu_admission_wait_ms", 0.0)
            ),
            "prompt_text_cpu_backpressure_wait_ms": float(
                (cpu_stage.prompt_cpu_profiled.profile or {}).get("text_cpu_backpressure_wait_ms", 0.0)
            ),
            "prompt_text_cpu_capacity_wait_ms": float(
                (cpu_stage.prompt_cpu_profiled.profile or {}).get("text_cpu_capacity_wait_ms", 0.0)
            ),
            "prompt_text_feature_queue_ms": prompt_feature_profiled.queue_ms,
            "prompt_text_feature_run_ms": prompt_feature_profiled.run_ms,
            "text_cpu_queue_ms": cpu_stage.target_cpu_profiled.queue_ms,
            "text_cpu_run_ms": cpu_stage.target_cpu_profiled.run_ms,
            "text_cpu_admission_wait_ms": float(
                (cpu_stage.target_cpu_profiled.profile or {}).get("text_cpu_admission_wait_ms", 0.0)
            ),
            "text_cpu_backpressure_wait_ms": float(
                (cpu_stage.target_cpu_profiled.profile or {}).get("text_cpu_backpressure_wait_ms", 0.0)
            ),
            "text_cpu_capacity_wait_ms": float(
                (cpu_stage.target_cpu_profiled.profile or {}).get("text_cpu_capacity_wait_ms", 0.0)
            ),
            "text_feature_queue_ms": target_feature_profiled.queue_ms,
            "text_feature_run_ms": target_feature_profiled.run_ms,
            "ref_audio_task_queue_ms": ref_audio_profiled.queue_ms,
            "ref_audio_task_run_ms": ref_audio_profiled.run_ms,
            "worker_prepare_inflight_on_enter": float(cpu_stage.current_inflight),
            "worker_prepare_peak_inflight": float(cpu_stage.peak_inflight),
        }
        if extra_profile:
            profile_overrides.update({key: float(value) for key, value in extra_profile.items()})
        ref_audio_bundle = dict(ref_audio_profiled.result)
        ref_audio_profile = dict(ref_audio_bundle.get("profile", {}))
        if ref_spec_result is not None:
            refer_spec, ref_spec_profiled = ref_spec_result
            ref_audio_bundle["refer_spec"] = refer_spec
            ref_audio_profile.update(
                {
                    "ref_spec_wait_ms": float(ref_spec_profiled.get("ref_spec_wait_ms", 0.0)),
                    "ref_spec_ms": float(ref_spec_profiled.get("ref_spec_ms", 0.0)),
                    "ref_spec_to_device_ms": float(ref_spec_profiled.get("ref_spec_to_device_ms", 0.0)),
                    "ref_spec_main_resample_ms": float(ref_spec_profiled.get("ref_spec_main_resample_ms", 0.0)),
                    "ref_spec_norm_ms": float(ref_spec_profiled.get("ref_spec_norm_ms", 0.0)),
                    "ref_spec_spectrogram_ms": float(ref_spec_profiled.get("ref_spec_spectrogram_ms", 0.0)),
                    "ref_spec_post_resample_ms": float(ref_spec_profiled.get("ref_spec_post_resample_ms", 0.0)),
                }
            )
        else:
            ref_audio_bundle["refer_spec"] = None
            ref_audio_profile.setdefault("ref_spec_wait_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_to_device_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_main_resample_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_norm_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_spectrogram_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_post_resample_ms", 0.0)
        ref_audio_bundle["profile"] = ref_audio_profile
        state = build_request_state_from_parts(
            tts=self.tts,
            spec=cpu_stage.spec,
            prompt_text=cpu_stage.prompt_text,
            text=cpu_stage.text,
            prompt_result=prompt_feature_profiled.result,
            target_result=target_feature_profiled.result,
            ref_audio_bundle=ref_audio_bundle,
            prepare_start=cpu_stage.prepare_start,
            prepare_sync_start=cpu_stage.prepare_start,
            profile_overrides=profile_overrides,
        )
        prepare_exec_finished_at = time.perf_counter()
        state.prepare_profile["executor_run_wall_ms"] = max(0.0, (prepare_exec_finished_at - cpu_stage.prepare_start) * 1000.0)
        return state, cpu_stage.prepare_start, prepare_exec_finished_at

    async def prepare_ref_spec_stages_async(
        self,
        phase_ones: list[Dict[str, Any]],
    ) -> list[tuple[tuple[Any, Any], Dict[str, float]] | Exception]:
        async def _one(phase_one: Dict[str, Any]):
            ref_audio_profiled = phase_one["ref_audio_profiled"]
            raw_audio = ref_audio_profiled.result["raw_audio"]
            raw_sr = int(ref_audio_profiled.result["raw_sr"])
            profiled = await self._run_ref_spec_stage(raw_audio, raw_sr)
            refer_spec, profile = profiled.result
            merged_profile = dict(profile)
            merged_profile["ref_spec_wait_ms"] = float(profiled.queue_ms)
            merged_profile["ref_spec_ms"] = float(profiled.run_ms)
            return refer_spec, merged_profile

        if not phase_ones:
            return []
        return list(await asyncio.gather(*[_one(phase_one) for phase_one in phase_ones], return_exceptions=True))

    def apply_ref_spec_result_to_state(
        self,
        state: T2SRequestState,
        ref_spec_result: tuple[tuple[Any, Any], Dict[str, float]],
    ) -> None:
        refer_spec, profile = ref_spec_result
        state.refer_spec = refer_spec
        state.prepare_profile["ref_spec_wait_ms"] = float(profile.get("ref_spec_wait_ms", 0.0))
        state.prepare_profile["ref_spec_ms"] = float(profile.get("ref_spec_ms", 0.0))
        state.prepare_profile["ref_spec_to_device_ms"] = float(profile.get("ref_spec_to_device_ms", 0.0))
        state.prepare_profile["ref_spec_main_resample_ms"] = float(profile.get("ref_spec_main_resample_ms", 0.0))
        state.prepare_profile["ref_spec_norm_ms"] = float(profile.get("ref_spec_norm_ms", 0.0))
        state.prepare_profile["ref_spec_spectrogram_ms"] = float(profile.get("ref_spec_spectrogram_ms", 0.0))
        state.prepare_profile["ref_spec_post_resample_ms"] = float(profile.get("ref_spec_post_resample_ms", 0.0))

    async def prepare_gpu_stages_profiled_async(
        self,
        cpu_stages: list[PreparedCpuStage],
    ) -> list[tuple[T2SRequestState, float, float] | Exception]:
        if not cpu_stages:
            return []
        if len(cpu_stages) == 1:
            single_stage = cpu_stages[0]
            try:
                return [await self.prepare_gpu_stage_profiled_async(single_stage)]
            except Exception as exc:  # noqa: PERF203
                return [exc]

        phase_one_started_at = time.perf_counter()
        phase_one_results = await asyncio.gather(
            *[self._prepare_gpu_phase_one(cpu_stage) for cpu_stage in cpu_stages],
            return_exceptions=True,
        )
        phase_one_finished_at = time.perf_counter()
        phase_one_wall_ms = max(0.0, (phase_one_finished_at - phase_one_started_at) * 1000.0)

        outputs: list[tuple[T2SRequestState, float, float] | Exception | None] = [None] * len(cpu_stages)
        pending_phase_two: list[tuple[int, PreparedCpuStage, Dict[str, Any]]] = []
        for index, (cpu_stage, phase_one) in enumerate(zip(cpu_stages, phase_one_results)):
            if isinstance(phase_one, Exception):
                outputs[index] = phase_one
                self._release_split_stage_slot()
                continue
            pending_phase_two.append((index, cpu_stage, phase_one))

        phase_two_started_at = time.perf_counter()
        phase_two_results = await asyncio.gather(
            *[self._prepare_gpu_phase_two(cpu_stage, phase_one) for _, cpu_stage, phase_one in pending_phase_two],
            return_exceptions=True,
        )
        phase_two_finished_at = time.perf_counter()
        phase_two_wall_ms = max(0.0, (phase_two_finished_at - phase_two_started_at) * 1000.0)

        for (index, cpu_stage, phase_one), phase_two in zip(pending_phase_two, phase_two_results):
            try:
                if isinstance(phase_two, Exception):
                    outputs[index] = phase_two
                    continue
                outputs[index] = self._build_gpu_prepare_result(
                    cpu_stage,
                    phase_one,
                    phase_two,
                    extra_profile={
                        "engine_prepare_audio_phase_mode": 1.0,
                        "engine_prepare_audio_phase_wall_ms": float(phase_one_wall_ms),
                        "engine_prepare_audio_phase_batch_size": float(len(cpu_stages)),
                        "engine_prepare_text_phase_wall_ms": float(phase_two_wall_ms),
                        "engine_prepare_text_phase_batch_size": float(len(pending_phase_two)),
                    },
                )
            except Exception as exc:  # noqa: PERF203
                outputs[index] = exc
            finally:
                self._release_split_stage_slot()

        return [item if item is not None else RuntimeError("prepare batch result missing") for item in outputs]

    async def prepare_gpu_audio_phases_async(
        self,
        cpu_stages: list[PreparedCpuStage],
    ) -> list[Dict[str, Any] | Exception]:
        if not cpu_stages:
            return []
        return list(
            await asyncio.gather(
                *[self._prepare_gpu_phase_one(cpu_stage) for cpu_stage in cpu_stages],
                return_exceptions=True,
            )
        )

    async def prepare_gpu_text_phases_async(
        self,
        items: list[tuple[PreparedCpuStage, Dict[str, Any]]],
    ) -> list[Dict[str, Any] | Exception]:
        if not items:
            return []
        return list(
            await asyncio.gather(
                *[self._prepare_gpu_phase_two(cpu_stage, phase_one) for cpu_stage, phase_one in items],
                return_exceptions=True,
            )
        )

    def build_gpu_prepare_result_from_phases(
        self,
        cpu_stage: PreparedCpuStage,
        phase_one: Dict[str, Any],
        phase_two: Dict[str, Any],
        extra_profile: Dict[str, float] | None = None,
    ) -> tuple[T2SRequestState, float, float]:
        try:
            return self._build_gpu_prepare_result(cpu_stage, phase_one, phase_two, extra_profile=extra_profile)
        finally:
            self._release_split_stage_slot()

    async def prepare_state_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> tuple[T2SRequestState, float, float]:
        cpu_stage = await self.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)
        return await self.prepare_gpu_stage_profiled_async(cpu_stage)
