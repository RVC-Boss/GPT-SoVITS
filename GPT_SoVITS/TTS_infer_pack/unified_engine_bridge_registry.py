from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SFinishedItem
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineRequestState, EngineStatus, SchedulerFinalizeTask, SchedulerPendingJob


class EngineRegistryBridgeFacade:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    @property
    def request_registry(self):
        return self.owner.request_registry

    @property
    def engine_prepare_queue_owner(self):
        return self.owner.engine_prepare_queue_owner

    @property
    def engine_prepare_text_queue_owner(self):
        return self.owner.engine_prepare_text_queue_owner

    @property
    def engine_prepare_ref_spec_queue_owner(self):
        return self.owner.engine_prepare_ref_spec_queue_owner

    @property
    def engine_finalize_queue_owner(self):
        return self.owner.engine_finalize_queue_owner

    @property
    def engine_dispatch_queue_owner(self):
        return self.owner.engine_dispatch_queue_owner

    @property
    def engine_decode_runtime_owner(self):
        return self.owner.engine_decode_runtime_owner

    @property
    def engine_job_registry(self):
        return self.owner.engine_job_registry

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
    ) -> EngineRequestState:
        return self.request_registry.register(
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
        self.request_registry.update(request_id, status, extra)

    def _merge_request_state_profile(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.request_registry.merge_profile(request_id, extra)

    def _complete_request_state(self, request_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.request_registry.complete(request_id, extra)

    def _fail_request_state(self, request_id: str, error: str) -> None:
        self.request_registry.fail(request_id, error)

    def _snapshot_request_registry(self) -> Dict[str, Any]:
        return self.request_registry.snapshot()

    def _snapshot_engine_prepare_state(self) -> Dict[str, Any]:
        audio_snapshot = self.engine_prepare_queue_owner.snapshot(max_request_ids=16)
        text_snapshot = self.engine_prepare_text_queue_owner.snapshot(max_request_ids=16)
        ref_spec_snapshot = self.engine_prepare_ref_spec_queue_owner.snapshot(max_request_ids=16)
        return {
            "waiting_count": int(audio_snapshot.get("waiting_count", 0))
            + int(text_snapshot.get("waiting_count", 0))
            + int(ref_spec_snapshot.get("waiting_count", 0)),
            "audio_waiting_count": int(audio_snapshot.get("waiting_count", 0)),
            "text_waiting_count": int(text_snapshot.get("waiting_count", 0)),
            "ref_spec_waiting_count": int(ref_spec_snapshot.get("waiting_count", 0)),
            "audio_waiting_request_ids": list(audio_snapshot.get("waiting_request_ids", [])),
            "text_waiting_request_ids": list(text_snapshot.get("waiting_request_ids", [])),
            "ref_spec_waiting_request_ids": list(ref_spec_snapshot.get("waiting_request_ids", [])),
            "peak_waiting": int(
                max(
                    int(audio_snapshot.get("peak_waiting", 0)),
                    int(text_snapshot.get("peak_waiting", 0)),
                    int(ref_spec_snapshot.get("peak_waiting", 0)),
                )
            ),
            "total_submitted": int(audio_snapshot.get("total_submitted", 0)),
            "total_completed": int(audio_snapshot.get("total_completed", 0)),
            "text_total_submitted": int(text_snapshot.get("total_submitted", 0)),
            "text_total_completed": int(text_snapshot.get("total_completed", 0)),
            "ref_spec_total_submitted": int(ref_spec_snapshot.get("total_submitted", 0)),
            "ref_spec_total_completed": int(ref_spec_snapshot.get("total_completed", 0)),
        }

    def _snapshot_engine_finalize_state(self) -> Dict[str, Any]:
        return self.engine_finalize_queue_owner.snapshot(max_request_ids=16)

    def _snapshot_engine_dispatch_state(self) -> Dict[str, Any]:
        return self.engine_dispatch_queue_owner.snapshot(
            max_request_ids=16,
            extra={"last_policy_snapshot": dict(self.owner.engine_dispatch_last_snapshot or {})},
        )

    def _register_engine_job(self, job: SchedulerPendingJob) -> None:
        self.engine_job_registry.register(job, keep_job=True)

    def _get_engine_job(self, request_id: str) -> SchedulerPendingJob | None:
        return self.engine_job_registry.get(request_id)

    def _pop_engine_job(self, request_id: str) -> SchedulerPendingJob | None:
        return self.engine_job_registry.pop(request_id)

    def _snapshot_engine_job_registry(self) -> Dict[str, Any]:
        return self.engine_job_registry.snapshot(max_request_ids=32)

    def _is_engine_drained(self) -> bool:
        prepare_empty = self.engine_prepare_queue_owner.is_drained()
        prepare_text_empty = self.engine_prepare_text_queue_owner.is_drained()
        prepare_ref_spec_empty = self.engine_prepare_ref_spec_queue_owner.is_drained()
        dispatch_empty = self.engine_dispatch_queue_owner.is_drained()
        finalize_empty = self.engine_finalize_queue_owner.is_drained()
        decode_pending_empty = not self.engine_decode_runtime_owner.has_pending_jobs()
        job_empty = self.engine_job_registry.is_empty()
        worker_state = self.scheduler_worker.snapshot()
        return bool(
            prepare_empty
            and prepare_text_empty
            and prepare_ref_spec_empty
            and dispatch_empty
            and finalize_empty
            and decode_pending_empty
            and job_empty
            and self.engine_decode_runtime_owner.get_active_batch() is None
            and int(worker_state.get("prepare_inflight", 0)) <= 0
            and int(worker_state.get("finalize_inflight", 0)) <= 0
            and int(worker_state.get("finalize_pending", 0)) <= 0
        )

    def _record_engine_job_done(self, request_id: str) -> None:
        self.engine_job_registry.mark_finished_and_remove(request_id)
        self.scheduler_worker.record_external_job_done(request_id)

    def _complete_engine_job(
        self,
        job: SchedulerPendingJob,
        item: T2SFinishedItem,
        *,
        sample_rate: int,
        audio_data: np.ndarray,
    ) -> None:
        completion_bridge = self.scheduler_worker.completion_bridge
        completion_bridge.build_completed_job_result(job, item, sample_rate=sample_rate, audio_data=audio_data)
        completion_bridge.complete_job(
            job,
            runtime_request_id=job.engine_request_id,
            runtime_extra=completion_bridge.build_runtime_complete_payload(job, item, sample_rate=sample_rate),
            on_job_finished=lambda rid=item.request_id: self._record_engine_job_done(rid),
        )

    def _fail_engine_jobs(self, request_ids: List[str], error: str) -> None:
        if not request_ids:
            return
        completion_bridge = self.scheduler_worker.completion_bridge
        for request_id in request_ids:
            job = self._get_engine_job(request_id)
            if job is None:
                continue
            completion_bridge.fail_job(
                job,
                error=error,
                on_job_finished=lambda rid=request_id: self._record_engine_job_done(rid),
            )

    def _add_engine_prefill_time(self, jobs: List[SchedulerPendingJob], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        for job in jobs:
            job.prefill_ms += delta_ms

    def _add_engine_merge_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        for request_id in request_ids:
            job = self._get_engine_job(request_id)
            if job is not None:
                job.merge_ms += delta_ms

    def _add_engine_decode_time(self, request_ids: List[str], elapsed_s: float) -> None:
        delta_ms = float(elapsed_s) * 1000.0
        activate_request_ids: List[str] = []
        for request_id in request_ids:
            job = self._get_engine_job(request_id)
            if job is None:
                continue
            if job.decode_steps == 0:
                activate_request_ids.append(job.engine_request_id)
            job.decode_ms += delta_ms
            job.decode_steps += 1
        for engine_request_id in activate_request_ids:
            self._update_request_state(engine_request_id, EngineStatus.ACTIVE_DECODE, None)

    def _enqueue_engine_finished_items(self, items: List[T2SFinishedItem]) -> None:
        if not items:
            return
        enqueued_at = time.perf_counter()
        tasks = [SchedulerFinalizeTask(request_id=item.request_id, item=item, enqueued_time=enqueued_at) for item in items]
        self.owner.engine_stage_coordinator.enqueue_worker_finished_for_finalize(tasks)
