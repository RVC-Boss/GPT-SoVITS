from __future__ import annotations


class EngineDecodeStageMixin:
    def run_engine_decode_runtime_once(self) -> bool:
        if not self.scheduler_worker.is_engine_decode_control_enabled():
            return False
        runtime_state = self.snapshot_engine_decode_runtime_state()
        pending_jobs = self.decode_runtime_owner.take_pending_jobs_nonblocking(
            wait_for_batch=int(runtime_state.get("active_request_count", 0)) <= 0
        )
        result = self.scheduler_worker.execute_decode_cycle(
            pending_jobs=pending_jobs,
            active_batch=self.decode_runtime_owner.get_active_batch(),
            external_bookkeeping=True,
        )
        prefill_phase = dict(result.get("prefill_phase") or {})
        if prefill_phase.get("error"):
            self.fail_engine_jobs(list(prefill_phase.get("error_request_ids") or []), str(prefill_phase.get("error")))
        else:
            prefill_jobs = list(prefill_phase.get("pending_jobs") or [])
            self.add_engine_prefill_time(prefill_jobs, float(prefill_phase.get("prefill_elapsed_s", 0.0)))
            self.add_engine_merge_time(
                [] if result.get("active_batch") is None else list(result["active_batch"].request_ids),
                float(prefill_phase.get("merge_elapsed_s", 0.0)),
            )
            self.enqueue_engine_finished_items(list(prefill_phase.get("finished_items") or []))
        decode_phase = dict(result.get("decode_phase") or {})
        if decode_phase.get("error"):
            self.fail_engine_jobs(list(decode_phase.get("error_request_ids") or []), str(decode_phase.get("error")))
        else:
            self.add_engine_decode_time(
                list(decode_phase.get("request_ids") or []),
                float(decode_phase.get("decode_elapsed_s", 0.0)),
            )
            self.enqueue_engine_finished_items(list(decode_phase.get("finished_items") or []))
        self.decode_runtime_owner.set_active_batch(result.get("active_batch"))
        if result.get("executed", False):
            self.decode_runtime_owner.refresh_state("engine_decode_cycle")
        return bool(result.get("executed", False))
