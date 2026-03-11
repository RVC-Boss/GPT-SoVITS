from __future__ import annotations

import os
import threading
from typing import Any

from GPT_SoVITS.TTS_infer_pack.unified_engine_api import EngineApiFacade
from GPT_SoVITS.TTS_infer_pack.unified_engine_bridge import EngineBridgeFacade
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import (
    EngineArbiterConfig,
    EngineDecodeRuntimeOwner,
    EnginePolicyArbiterController,
    EnginePolicyConfig,
    EngineRequestRegistry,
    EngineTaskQueueOwner,
    ModelRegistry,
    ReferenceRegistry,
    RuntimeStateCallbacks,
    SchedulerJobRegistry,
)
from GPT_SoVITS.TTS_infer_pack.unified_engine_runtime import EngineRuntimeFacade
from GPT_SoVITS.TTS_infer_pack.unified_engine_stage import EngineStageCoordinator
from GPT_SoVITS.TTS_infer_pack.unified_engine_worker import UnifiedSchedulerWorker


class EngineCompositionBuilder:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def build(self, *, max_steps: int, micro_batch_wait_ms: int) -> None:
        self._init_registries_and_locks()
        self._init_worker(max_steps=max_steps, micro_batch_wait_ms=micro_batch_wait_ms)
        self._init_policy_configs(micro_batch_wait_ms=micro_batch_wait_ms)
        self._init_runtime_owners()
        self._init_stage_coordinator()
        self._init_arbiter()
        self._init_facades()
        self._start_arbiter_thread()

    def _init_registries_and_locks(self) -> None:
        owner = self.owner
        owner.reference_registry = ReferenceRegistry()
        owner.model_registry = ModelRegistry(
            t2s_weights_path=str(owner.tts.configs.t2s_weights_path),
            vits_weights_path=str(owner.tts.configs.vits_weights_path),
        )
        owner.request_registry = EngineRequestRegistry(
            recent_limit=max(1, int(os.environ.get("GPTSOVITS_ENGINE_RECENT_REQUEST_LIMIT", "64")))
        )
        owner.engine_job_registry = SchedulerJobRegistry(threading.Lock())
        owner.direct_tts_lock = threading.RLock()
        owner.management_lock = threading.RLock()
        owner.engine_dispatch_last_snapshot = {}

    def _init_worker(self, *, max_steps: int, micro_batch_wait_ms: int) -> None:
        owner = self.owner
        owner.scheduler_worker = UnifiedSchedulerWorker(
            owner.tts,
            max_steps=max_steps,
            micro_batch_wait_ms=micro_batch_wait_ms,
            runtime_callbacks=RuntimeStateCallbacks(
                update=owner._update_request_state,
                complete=owner._complete_request_state,
                fail=owner._fail_request_state,
                decode_runtime_update=owner._update_engine_decode_runtime_state,
            ),
            external_finalize_submit=owner._enqueue_worker_finished_for_finalize,
        )

    def _init_policy_configs(self, *, micro_batch_wait_ms: int) -> None:
        owner = self.owner
        worker_capacity_limits = owner.scheduler_worker.get_capacity_limits()
        prepare_max_inflight = int(owner.scheduler_worker.get_prepare_max_inflight())
        owner.engine_policy_config = EnginePolicyConfig(
            enabled=owner._env_flag("GPTSOVITS_ENGINE_POLICY_ENABLE", True),
            poll_wait_ms=max(1.0, owner._env_float("GPTSOVITS_ENGINE_POLICY_POLL_WAIT_MS", float(micro_batch_wait_ms))),
            decode_backlog_soft_max=max(
                0,
                owner._env_int(
                    "GPTSOVITS_ENGINE_POLICY_DECODE_BACKLOG_SOFT_MAX",
                    int(worker_capacity_limits["decode_backlog_max"]),
                ),
            ),
            finalize_pending_soft_max=max(
                0,
                owner._env_int(
                    "GPTSOVITS_ENGINE_POLICY_FINALIZE_PENDING_SOFT_MAX",
                    int(worker_capacity_limits["finalize_pending_max"]),
                ),
            ),
            prepare_inflight_soft_max=max(
                0,
                owner._env_int("GPTSOVITS_ENGINE_POLICY_PREPARE_INFLIGHT_SOFT_MAX", prepare_max_inflight),
            ),
            active_decode_soft_max=max(0, owner._env_int("GPTSOVITS_ENGINE_POLICY_ACTIVE_DECODE_SOFT_MAX", 0)),
            ready_for_prefill_soft_max=max(0, owner._env_int("GPTSOVITS_ENGINE_POLICY_READY_FOR_PREFILL_SOFT_MAX", 0)),
            active_request_soft_max=max(0, owner._env_int("GPTSOVITS_ENGINE_POLICY_ACTIVE_REQUEST_SOFT_MAX", 0)),
        )
        owner.engine_arbiter_config = EngineArbiterConfig(
            poll_wait_ms=max(1.0, owner._env_float("GPTSOVITS_ENGINE_ARBITER_POLL_WAIT_MS", float(micro_batch_wait_ms))),
            decode_burst=max(1, owner._env_int("GPTSOVITS_ENGINE_ARBITER_DECODE_BURST", 4)),
            prepare_aging_ms=max(0.0, owner._env_float("GPTSOVITS_ENGINE_ARBITER_PREPARE_AGING_MS", 10.0)),
            finalize_aging_ms=max(0.0, owner._env_float("GPTSOVITS_ENGINE_ARBITER_FINALIZE_AGING_MS", 10.0)),
        )

    def _init_runtime_owners(self) -> None:
        owner = self.owner
        owner.engine_decode_runtime_owner = EngineDecodeRuntimeOwner(
            get_decode_runtime_counters=owner.scheduler_worker.get_decode_runtime_counters,
            get_micro_batch_wait_s=owner.scheduler_worker.get_micro_batch_wait_s,
        )
        owner.engine_prepare_queue_owner = EngineTaskQueueOwner(completion_key="total_completed")
        owner.engine_finalize_queue_owner = EngineTaskQueueOwner(completion_key="total_completed")
        owner.engine_dispatch_queue_owner = EngineTaskQueueOwner(completion_key="total_dispatched")

    def _init_stage_coordinator(self) -> None:
        owner = self.owner
        owner.engine_stage_coordinator = EngineStageCoordinator(
            tts=owner.tts,
            scheduler_worker=owner.scheduler_worker,
            prepare_queue_owner=owner.engine_prepare_queue_owner,
            finalize_queue_owner=owner.engine_finalize_queue_owner,
            dispatch_queue_owner=owner.engine_dispatch_queue_owner,
            decode_runtime_owner=owner.engine_decode_runtime_owner,
            update_request_state=owner._update_request_state,
            merge_request_state_profile=owner._merge_request_state_profile,
            fail_request_state=owner._fail_request_state,
            get_engine_job=owner._get_engine_job,
            register_engine_job=owner._register_engine_job,
            fail_engine_jobs=owner._fail_engine_jobs,
            complete_engine_job=owner._complete_engine_job,
            add_engine_prefill_time=owner._add_engine_prefill_time,
            add_engine_merge_time=owner._add_engine_merge_time,
            add_engine_decode_time=owner._add_engine_decode_time,
            enqueue_engine_finished_items=owner._enqueue_engine_finished_items,
            snapshot_engine_dispatch_state=owner._snapshot_engine_dispatch_state,
            snapshot_engine_decode_runtime_state=owner._snapshot_engine_decode_runtime_state,
        )

    def _init_arbiter(self) -> None:
        owner = self.owner
        owner.engine_policy_arbiter = EnginePolicyArbiterController(
            policy_config=owner.engine_policy_config,
            arbiter_config=owner.engine_arbiter_config,
            snapshot_request_registry=owner._snapshot_request_registry,
            get_worker_state=owner.scheduler_worker.snapshot,
            snapshot_prepare_state=owner._snapshot_engine_prepare_state,
            snapshot_finalize_state=owner._snapshot_engine_finalize_state,
            snapshot_dispatch_state=owner._snapshot_engine_dispatch_state,
            snapshot_decode_runtime_state=owner._snapshot_engine_decode_runtime_state,
            snapshot_job_registry=owner._snapshot_engine_job_registry,
            peek_queue_age_ms=owner.engine_stage_coordinator.peek_queue_age_ms,
            merge_request_state_profile=owner._merge_request_state_profile,
        )
        owner.engine_stage_coordinator.bind_arbiter(
            notify_arbiter=owner._notify_engine_arbiter,
            select_stage=owner._select_engine_stage,
            mark_arbiter_tick=lambda stage, reason, policy_allowed: owner._mark_arbiter_tick(
                stage=stage,
                reason=reason,
                policy_allowed=policy_allowed,
            ),
            wait_arbiter=owner.engine_policy_arbiter.wait,
        )

    def _init_facades(self) -> None:
        owner = self.owner
        owner.bridge_facade = EngineBridgeFacade(owner)
        owner.api_facade = EngineApiFacade(owner)
        owner.runtime_facade = EngineRuntimeFacade(owner)

    def _start_arbiter_thread(self) -> None:
        owner = self.owner
        owner.engine_arbiter_thread = threading.Thread(
            target=owner._run_engine_arbiter_loop,
            name="unified-engine-arbiter",
            daemon=True,
        )
        owner.engine_arbiter_thread.start()
