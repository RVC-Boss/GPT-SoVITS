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
    build_request_state_from_parts,
    normalize_sentence,
)


@dataclass
class ProfiledResult:
    result: Any
    submit_at: float
    started_at: float
    finished_at: float

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
        self._inflight_semaphore = asyncio.Semaphore(self.max_inflight) if self.max_inflight > 0 else None
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
        ref_audio_default_workers = max(1, int(os.environ.get("GPTSOVITS_PREPARE_REF_SLOTS", "4")))
        self.ref_audio_workers = max(
            1,
            int(os.environ.get("GPTSOVITS_PREPARE_REF_ASYNC_WORKERS", str(ref_audio_default_workers))),
        )
        self.ref_audio_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.ref_audio_workers,
            thread_name_prefix="prepare-ref-audio",
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

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            return {
                "inflight": int(self.inflight),
                "peak_inflight": int(self.peak_inflight),
                "max_inflight": int(self.max_inflight),
                "text_feature_workers": int(self.text_feature_workers),
                "ref_audio_workers": int(self.ref_audio_workers),
            }

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

    def _build_text_features(self, prepared_segments, language: str, cpu_run_ms: float) -> PreparedTextFeatures:
        profile: Dict[str, float] = {"cpu_preprocess_ms": float(cpu_run_ms)}
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
        executor = getattr(self.tts, "prepare_text_cpu_executor", None)
        if executor is None:
            submit_at = time.perf_counter()
            return self._run_profiled(self._prepare_text_cpu, submit_at, text, language)
        return await self._run_on_executor(executor, self._prepare_text_cpu, text, language)

    async def _run_text_feature_stage(self, prepared_segments, language: str, cpu_run_ms: float) -> ProfiledResult:
        return await self._run_on_executor(self.text_feature_executor, self._build_text_features, prepared_segments, language, cpu_run_ms)

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
    ) -> tuple[ProfiledResult, ProfiledResult]:
        if self.text_feature_executor is not None:
            prompt_feature_task = asyncio.create_task(
                self._run_text_feature_stage(prompt_segments, None, prompt_cpu_run_ms)
            )
            target_feature_task = asyncio.create_task(
                self._run_text_feature_stage(target_segments, None, target_cpu_run_ms)
            )
            return await asyncio.gather(prompt_feature_task, target_feature_task)

        prompt_profile: Dict[str, float] = {"cpu_preprocess_ms": float(prompt_cpu_run_ms)}
        target_profile: Dict[str, float] = {"cpu_preprocess_ms": float(target_cpu_run_ms)}
        submit_at = time.perf_counter()
        started_at = float(submit_at)
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

    async def _run_ref_audio_stage(self, ref_audio_path: str) -> ProfiledResult:
        return await self._run_on_executor(self.ref_audio_executor, self.tts.extract_ref_audio_bundle, ref_audio_path)

    def _release_split_stage_slot(self) -> None:
        self._mark_leave()
        if self._inflight_semaphore is not None:
            self._inflight_semaphore.release()

    async def prepare_cpu_stage_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> PreparedCpuStage:
        admission_start = time.perf_counter()
        if self._inflight_semaphore is not None:
            await self._inflight_semaphore.acquire()
        prepare_admission_wait_ms = max(0.0, (time.perf_counter() - admission_start) * 1000.0)
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
            text_pair_start = time.perf_counter()
            ref_audio_task = asyncio.create_task(self._run_ref_audio_stage(str(cpu_stage.spec.ref_audio_path)))
            text_feature_pair_task = asyncio.create_task(
                self._run_text_feature_pair_stage(
                    cpu_stage.prompt_cpu_profiled.result,
                    cpu_stage.target_cpu_profiled.result,
                    cpu_stage.prompt_cpu_profiled.run_ms,
                    cpu_stage.target_cpu_profiled.run_ms,
                )
            )
            (prompt_feature_profiled, target_feature_profiled), ref_audio_profiled = await asyncio.gather(
                text_feature_pair_task,
                ref_audio_task,
            )
            text_pair_end = time.perf_counter()
            state = build_request_state_from_parts(
                tts=self.tts,
                spec=cpu_stage.spec,
                prompt_text=cpu_stage.prompt_text,
                text=cpu_stage.text,
                prompt_result=prompt_feature_profiled.result,
                target_result=target_feature_profiled.result,
                ref_audio_bundle=ref_audio_profiled.result,
                prepare_start=cpu_stage.prepare_start,
                prepare_sync_start=cpu_stage.prepare_start,
                profile_overrides={
                    "executor_queue_ms": max(0.0, (cpu_stage.prepare_start - cpu_stage.prepare_submit_at) * 1000.0),
                    "prepare_admission_wait_ms": cpu_stage.prepare_admission_wait_ms,
                    "executor_run_wall_ms": max(0.0, (time.perf_counter() - cpu_stage.prepare_start) * 1000.0),
                    "text_feature_pair_ms": max(0.0, (text_pair_end - text_pair_start) * 1000.0),
                    "prompt_text_parallel_future_wait_ms": 0.0,
                    "prompt_text_parallel_future_executor_queue_ms": 0.0,
                    "prompt_text_parallel_future_run_ms": 0.0,
                    "prompt_text_parallel_future_finish_after_submit_ms": 0.0,
                    "prompt_text_parallel_future_queue_tail_after_target_ms": 0.0,
                    "prompt_text_parallel_future_run_tail_after_target_ms": 0.0,
                    "prompt_text_cpu_queue_ms": cpu_stage.prompt_cpu_profiled.queue_ms,
                    "prompt_text_cpu_run_ms": cpu_stage.prompt_cpu_profiled.run_ms,
                    "prompt_text_feature_queue_ms": prompt_feature_profiled.queue_ms,
                    "prompt_text_feature_run_ms": prompt_feature_profiled.run_ms,
                    "text_cpu_queue_ms": cpu_stage.target_cpu_profiled.queue_ms,
                    "text_cpu_run_ms": cpu_stage.target_cpu_profiled.run_ms,
                    "text_feature_queue_ms": target_feature_profiled.queue_ms,
                    "text_feature_run_ms": target_feature_profiled.run_ms,
                    "ref_audio_task_queue_ms": ref_audio_profiled.queue_ms,
                    "ref_audio_task_run_ms": ref_audio_profiled.run_ms,
                    "worker_prepare_inflight_on_enter": float(cpu_stage.current_inflight),
                    "worker_prepare_peak_inflight": float(cpu_stage.peak_inflight),
                },
            )
            prepare_exec_finished_at = time.perf_counter()
            state.prepare_profile["executor_run_wall_ms"] = max(
                0.0, (prepare_exec_finished_at - cpu_stage.prepare_start) * 1000.0
            )
            return state, cpu_stage.prepare_start, prepare_exec_finished_at
        finally:
            self._release_split_stage_slot()

    async def prepare_state_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> tuple[T2SRequestState, float, float]:
        cpu_stage = await self.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)
        return await self.prepare_gpu_stage_profiled_async(cpu_stage)
