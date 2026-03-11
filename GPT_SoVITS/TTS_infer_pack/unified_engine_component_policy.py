from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from GPT_SoVITS.TTS_infer_pack.unified_engine_component_registry import EngineStatus


@dataclass
class EnginePolicyConfig:
    enabled: bool = True
    poll_wait_ms: float = 5.0
    decode_backlog_soft_max: int = 0
    finalize_pending_soft_max: int = 0
    prepare_inflight_soft_max: int = 0
    active_decode_soft_max: int = 0
    ready_for_prefill_soft_max: int = 0
    active_request_soft_max: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "poll_wait_ms": float(self.poll_wait_ms),
            "decode_backlog_soft_max": int(self.decode_backlog_soft_max),
            "finalize_pending_soft_max": int(self.finalize_pending_soft_max),
            "prepare_inflight_soft_max": int(self.prepare_inflight_soft_max),
            "active_decode_soft_max": int(self.active_decode_soft_max),
            "ready_for_prefill_soft_max": int(self.ready_for_prefill_soft_max),
            "active_request_soft_max": int(self.active_request_soft_max),
        }


@dataclass
class EngineArbiterConfig:
    poll_wait_ms: float = 5.0
    decode_burst: int = 4
    prepare_aging_ms: float = 10.0
    finalize_aging_ms: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "poll_wait_ms": float(self.poll_wait_ms),
            "decode_burst": int(self.decode_burst),
            "prepare_aging_ms": float(self.prepare_aging_ms),
            "finalize_aging_ms": float(self.finalize_aging_ms),
        }


@dataclass
class EngineArbiterState:
    total_ticks: int = 0
    total_idle_ticks: int = 0
    total_prepare_dispatches: int = 0
    total_decode_dispatches: int = 0
    total_decode_runtime_ticks: int = 0
    total_finalize_dispatches: int = 0
    decode_budget_remaining: int = 0
    last_stage: str = "idle"
    last_reason: str = "init"
    last_observed_at: float = 0.0
    last_policy_allowed: bool = True


class EnginePolicyArbiterController:
    def __init__(
        self,
        *,
        policy_config: EnginePolicyConfig,
        arbiter_config: EngineArbiterConfig,
        snapshot_request_registry: Callable[[], Dict[str, Any]],
        get_worker_state: Callable[[], Dict[str, Any]],
        snapshot_prepare_state: Callable[[], Dict[str, Any]],
        snapshot_finalize_state: Callable[[], Dict[str, Any]],
        snapshot_dispatch_state: Callable[[], Dict[str, Any]],
        snapshot_decode_runtime_state: Callable[[], Dict[str, Any]],
        snapshot_job_registry: Callable[[], Dict[str, Any]],
        peek_queue_age_ms: Callable[[str], float],
        merge_request_state_profile: Callable[[str, Optional[Dict[str, Any]]], None],
    ) -> None:
        self.policy_config = policy_config
        self.policy_poll_s = max(0.001, float(self.policy_config.poll_wait_ms) / 1000.0)
        self.arbiter_config = arbiter_config
        self.arbiter_poll_s = max(0.001, float(self.arbiter_config.poll_wait_ms) / 1000.0)
        self.condition = threading.Condition()
        self.state = EngineArbiterState(
            decode_budget_remaining=int(self.arbiter_config.decode_burst),
            last_observed_at=time.perf_counter(),
        )
        self.snapshot_request_registry = snapshot_request_registry
        self.get_worker_state = get_worker_state
        self.snapshot_prepare_state = snapshot_prepare_state
        self.snapshot_finalize_state = snapshot_finalize_state
        self.snapshot_dispatch_state = snapshot_dispatch_state
        self.snapshot_decode_runtime_state = snapshot_decode_runtime_state
        self.snapshot_job_registry = snapshot_job_registry
        self.peek_queue_age_ms = peek_queue_age_ms
        self.merge_request_state_profile = merge_request_state_profile

    def snapshot_state(self) -> Dict[str, Any]:
        with self.condition:
            return {
                "config": self.arbiter_config.to_dict(),
                "total_ticks": int(self.state.total_ticks),
                "total_idle_ticks": int(self.state.total_idle_ticks),
                "total_prepare_dispatches": int(self.state.total_prepare_dispatches),
                "total_decode_dispatches": int(self.state.total_decode_dispatches),
                "total_decode_runtime_ticks": int(self.state.total_decode_runtime_ticks),
                "total_finalize_dispatches": int(self.state.total_finalize_dispatches),
                "decode_budget_remaining": int(self.state.decode_budget_remaining),
                "last_stage": str(self.state.last_stage),
                "last_reason": str(self.state.last_reason),
                "last_policy_allowed": bool(self.state.last_policy_allowed),
                "last_observed_at": float(self.state.last_observed_at),
            }

    def notify(self) -> None:
        with self.condition:
            self.condition.notify_all()

    def wait(self) -> None:
        with self.condition:
            self.condition.wait(timeout=self.arbiter_poll_s)

    def mark_tick(self, *, stage: str, reason: str, policy_allowed: bool) -> None:
        with self.condition:
            self.state.total_ticks += 1
            if stage == "idle":
                self.state.total_idle_ticks += 1
            elif stage == "prepare":
                self.state.total_prepare_dispatches += 1
                self.state.decode_budget_remaining = int(self.arbiter_config.decode_burst)
            elif stage == "finalize":
                self.state.total_finalize_dispatches += 1
                self.state.decode_budget_remaining = int(self.arbiter_config.decode_burst)
            elif stage == "decode_dispatch":
                self.state.total_decode_dispatches += 1
            elif stage == "decode_runtime":
                self.state.total_decode_runtime_ticks += 1
                self.state.decode_budget_remaining = max(0, int(self.state.decode_budget_remaining) - 1)
            self.state.last_stage = str(stage)
            self.state.last_reason = str(reason)
            self.state.last_policy_allowed = bool(policy_allowed)
            self.state.last_observed_at = time.perf_counter()

    def build_stage_counters(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        prepare_dispatcher_state = self.snapshot_prepare_state()
        finalize_dispatcher_state = self.snapshot_finalize_state()
        dispatcher_state = self.snapshot_dispatch_state()
        active_requests = list(request_registry.get("active_requests", []))
        status_counts: Dict[str, int] = {}
        for item in active_requests:
            status = str(item.get("status", "UNKNOWN"))
            status_counts[status] = status_counts.get(status, 0) + 1

        worker_pending_jobs = int(worker_state.get("pending_jobs", 0))
        worker_decode_active_size = int(worker_state.get("running_requests", 0))
        worker_prepare_inflight = int(worker_state.get("prepare_inflight", 0))
        worker_finalize_pending = int(worker_state.get("finalize_pending", 0))
        worker_finalize_inflight = int(worker_state.get("finalize_inflight", 0))
        engine_decode_runtime_state = self.snapshot_decode_runtime_state()
        engine_job_registry = self.snapshot_job_registry()
        decode_runtime_pending_jobs = int(engine_decode_runtime_state.get("pending_jobs", 0))
        decode_runtime_active_size = int(engine_decode_runtime_state.get("active_request_count", 0))
        return {
            "active_request_count": int(len(active_requests)),
            "status_counts": status_counts,
            "queued_request_count": int(status_counts.get(EngineStatus.QUEUED, 0)),
            "cpu_prepare_request_count": int(status_counts.get(EngineStatus.CPU_PREPARING, 0)),
            "gpu_prepare_request_count": int(status_counts.get(EngineStatus.GPU_PREPARING, 0)),
            "ready_for_prefill_request_count": int(status_counts.get(EngineStatus.READY_FOR_PREFILL, 0)),
            "active_decode_request_count": int(status_counts.get(EngineStatus.ACTIVE_DECODE, 0)),
            "ready_for_finalize_request_count": int(status_counts.get(EngineStatus.READY_FOR_FINALIZE, 0)),
            "finalizing_request_count": int(status_counts.get(EngineStatus.FINALIZING, 0)),
            "streaming_request_count": int(status_counts.get(EngineStatus.STREAMING, 0)),
            "worker_pending_jobs": worker_pending_jobs,
            "worker_decode_active_size": worker_decode_active_size,
            "worker_decode_control_enabled": bool(worker_state.get("engine_decode_control_enabled", False)),
            "worker_decode_runtime_has_work": bool(worker_state.get("decode_runtime_has_work", False)),
            "engine_decode_runtime_pending_jobs": decode_runtime_pending_jobs,
            "engine_decode_runtime_active_request_count": decode_runtime_active_size,
            "engine_decode_runtime_has_work": bool(engine_decode_runtime_state.get("has_work", False)),
            "engine_job_registry_count": int(engine_job_registry.get("job_count", 0)),
            "worker_prepare_inflight": worker_prepare_inflight,
            "worker_finalize_pending": worker_finalize_pending,
            "worker_finalize_inflight": worker_finalize_inflight,
            "engine_gpu_prepare_queue_count": int(prepare_dispatcher_state.get("waiting_count", 0)),
            "engine_finalize_queue_count": int(finalize_dispatcher_state.get("waiting_count", 0)),
            "engine_decode_waiting_queue_count": int(dispatcher_state.get("waiting_count", 0)),
            "decode_backlog": int(
                decode_runtime_pending_jobs + decode_runtime_active_size
                if bool(worker_state.get("engine_decode_control_enabled", False))
                else worker_pending_jobs + worker_decode_active_size
            ),
        }

    def build_policy_snapshot(
        self,
        request_registry: Dict[str, Any],
        worker_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        counters = self.build_stage_counters(request_registry, worker_state)
        config = self.policy_config.to_dict()
        blocked_reasons: List[Dict[str, Any]] = []
        finalize_pending_total = int(counters["worker_finalize_pending"]) + int(counters.get("engine_finalize_queue_count", 0))
        limit_checks = [
            ("decode_backlog", counters["decode_backlog"], int(config["decode_backlog_soft_max"])),
            ("finalize_pending", finalize_pending_total, int(config["finalize_pending_soft_max"])),
            ("prepare_inflight", counters["worker_prepare_inflight"], int(config["prepare_inflight_soft_max"])),
            ("active_decode_requests", counters["active_decode_request_count"], int(config["active_decode_soft_max"])),
            ("ready_for_prefill_requests", counters["ready_for_prefill_request_count"], int(config["ready_for_prefill_soft_max"])),
            ("active_requests", counters["active_request_count"], int(config["active_request_soft_max"])),
        ]
        if bool(config["enabled"]):
            for name, value, limit in limit_checks:
                if limit > 0 and int(value) >= int(limit):
                    blocked_reasons.append({"metric": name, "value": int(value), "limit": int(limit)})
        return {
            "enabled": bool(config["enabled"]),
            "allowed": (not bool(config["enabled"])) or not blocked_reasons,
            "blocked_reasons": blocked_reasons,
            "config": config,
            "metrics": {
                "active_request_count": int(counters["active_request_count"]),
                "queued_request_count": int(counters["queued_request_count"]),
                "ready_for_prefill_request_count": int(counters["ready_for_prefill_request_count"]),
                "active_decode_request_count": int(counters["active_decode_request_count"]),
                "engine_gpu_prepare_queue_count": int(counters["engine_gpu_prepare_queue_count"]),
                "engine_decode_waiting_queue_count": int(counters["engine_decode_waiting_queue_count"]),
                "decode_backlog": int(counters["decode_backlog"]),
                "prepare_inflight": int(counters["worker_prepare_inflight"]),
                "finalize_pending": int(finalize_pending_total),
                "engine_finalize_queue_count": int(counters.get("engine_finalize_queue_count", 0)),
                "finalize_inflight": int(counters["worker_finalize_inflight"]),
            },
            "observed_at": time.perf_counter(),
        }

    async def wait_for_policy_admission(
        self,
        *,
        request_id: str | None,
        timeout_sec: float | None,
    ) -> tuple[float, Dict[str, Any]]:
        request_registry = self.snapshot_request_registry()
        worker_state = self.get_worker_state()
        snapshot = self.build_policy_snapshot(request_registry, worker_state)
        if not self.policy_config.enabled:
            return 0.0, snapshot
        start = time.perf_counter()
        deadline = None if timeout_sec in [None, ""] else (start + max(0.0, float(timeout_sec)))
        while True:
            request_registry = self.snapshot_request_registry()
            worker_state = self.get_worker_state()
            snapshot = self.build_policy_snapshot(request_registry, worker_state)
            if snapshot["allowed"]:
                wait_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
                if request_id not in [None, ""]:
                    self.merge_request_state_profile(
                        str(request_id),
                        {
                            "engine_policy_wait_ms": float(wait_ms),
                            "engine_policy_snapshot": snapshot,
                        },
                    )
                return wait_ms, snapshot
            now = time.perf_counter()
            if deadline is not None and now >= deadline:
                blocked_summary = ", ".join(
                    f"{item['metric']}={item['value']}/{item['limit']}" for item in snapshot.get("blocked_reasons", [])
                )
                raise TimeoutError(f"engine policy admission timeout ({blocked_summary})")
            await asyncio.sleep(self.policy_poll_s)

    def select_stage(self) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        request_registry = self.snapshot_request_registry()
        worker_state = self.get_worker_state()
        policy_snapshot = self.build_policy_snapshot(request_registry, worker_state)
        prepare_waiting = int(self.snapshot_prepare_state().get("waiting_count", 0))
        finalize_waiting = int(self.snapshot_finalize_state().get("waiting_count", 0))
        decode_waiting = int(self.snapshot_dispatch_state().get("waiting_count", 0))
        decode_runtime_state = self.snapshot_decode_runtime_state()
        worker_decode_has_work = bool(decode_runtime_state.get("has_work", False))
        worker_decode_control_enabled = bool(worker_state.get("engine_decode_control_enabled", False))
        worker_pending_jobs = int(decode_runtime_state.get("pending_jobs", 0))
        worker_running_requests = int(decode_runtime_state.get("active_request_count", 0))
        prepare_age_ms = float(self.peek_queue_age_ms("prepare"))
        finalize_age_ms = float(self.peek_queue_age_ms("finalize"))
        decode_runtime_pending_age_ms = float(self.peek_queue_age_ms("decode_runtime_pending"))
        decode_budget_remaining = int(self.snapshot_state().get("decode_budget_remaining", 0))
        policy_allowed = bool(policy_snapshot.get("allowed", True))
        if (
            worker_decode_control_enabled
            and worker_decode_has_work
            and policy_allowed
            and decode_budget_remaining > 0
            and (worker_running_requests > 0 or worker_pending_jobs > 0)
        ):
            return "decode_runtime", "worker_active_batch_progress", policy_snapshot, worker_state
        if (
            worker_decode_control_enabled
            and worker_pending_jobs > 0
            and policy_allowed
            and decode_runtime_pending_age_ms >= float(self.arbiter_config.prepare_aging_ms)
        ):
            return "decode_runtime", "decode_runtime_pending_aging", policy_snapshot, worker_state
        if (
            decode_waiting > 0
            and policy_allowed
            and (not worker_decode_control_enabled or not worker_decode_has_work or worker_pending_jobs <= 0)
        ):
            return "decode_dispatch", "dispatch_prepared_state", policy_snapshot, worker_state
        if finalize_waiting > 0 and (decode_waiting <= 0 or not policy_allowed or decode_budget_remaining <= 0):
            return "finalize", "decode_blocked_or_budget_exhausted", policy_snapshot, worker_state
        if finalize_waiting > 0 and finalize_age_ms >= float(self.arbiter_config.finalize_aging_ms):
            return "finalize", "finalize_aging", policy_snapshot, worker_state
        if prepare_waiting > 0 and (decode_waiting <= 0 or not policy_allowed or decode_budget_remaining <= 0):
            return "prepare", "decode_blocked_or_budget_exhausted", policy_snapshot, worker_state
        if prepare_waiting > 0 and prepare_age_ms >= float(self.arbiter_config.prepare_aging_ms):
            return "prepare", "prepare_aging", policy_snapshot, worker_state
        if worker_decode_control_enabled and worker_decode_has_work and policy_allowed:
            return "decode_runtime", "worker_active_batch_progress_fallback", policy_snapshot, worker_state
        if decode_waiting > 0 and policy_allowed:
            return "decode_dispatch", "decode_priority_fallback", policy_snapshot, worker_state
        if finalize_waiting > 0:
            return "finalize", "finalize_fallback", policy_snapshot, worker_state
        if prepare_waiting > 0:
            return "prepare", "prepare_fallback", policy_snapshot, worker_state
        return "idle", "no_pending_work", policy_snapshot, worker_state
