from __future__ import annotations

import os
import time
import warnings
from collections import deque
from collections.abc import MutableSequence
from typing import Optional, TypeVar

import torch

from tools.acceleration import cuda_graph_available
from ...logger import timer

from .sample_funcs import sample_naive
from .structs import KVCache, T2SRequest, T2SSession
from .t2s_model_abc import T2SDecoderABC, TorchProfiler


T = TypeVar("T")


def _slice_tensors(obj: T, bsz: int) -> T:
    if isinstance(obj, torch.Tensor):
        if obj.dim() > 0 and obj.size(0) >= bsz:
            return obj[:bsz]  # type: ignore
        return obj  # type: ignore
    if isinstance(obj, list):
        return [_slice_tensors(o, bsz) for o in obj]  # type: ignore
    if isinstance(obj, tuple):
        return tuple(_slice_tensors(o, bsz) for o in obj)  # type: ignore
    if isinstance(obj, dict):
        return {k: _slice_tensors(v, bsz) for k, v in obj.items()}  # type: ignore
    return obj


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mtia":
        torch.mtia.synchronize()
    elif device.type == "cpu":
        pass


class ModelRunner:
    """Model-side executor handling prefill, decode, and CUDA graph binding."""

    def __init__(
        self,
        decoder: T2SDecoderABC,
        device: torch.device,
        dtype: torch.dtype,
        waiting_bind: deque[T2SSession],
        use_cuda_graph: bool = True,
    ):
        self.max_batch_size = decoder.max_batch_size

        self.decoder = decoder
        self.device = device
        self.dtype = dtype
        self.graph_enabled = use_cuda_graph

        self.waiting_bind: deque[T2SSession] = waiting_bind
        self.slot_sessions: list[Optional[T2SSession]] = [None] * decoder.max_batch_size
        self.session_slots: dict[int, list[int]] = {}
        self.bound_sessions: list[T2SSession] = []
        self.sampler = sample_naive()

        extra_buffer_factory = decoder.extra_buffer_factory
        is_applicable = decoder.graph_applicable

        self.graph_applicable = bool(use_cuda_graph and cuda_graph_available(device) and is_applicable)

        dtype_bias = decoder.bert_proj.bias.dtype

        # Shared runtime buffers for both eager and graph replay
        self.xy_pos_buf = torch.zeros((self.max_batch_size, 1, decoder.embedding_dim), device=device, dtype=dtype_bias)
        self.xy_dec_buf = torch.zeros_like(self.xy_pos_buf)
        self.input_pos_buf = torch.ones((self.max_batch_size,), device=device, dtype=torch.int32)
        self.kv_cache_buf: MutableSequence[KVCache] = decoder.init_cache(self.max_batch_size)

        self.step_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.int32)
        self.prefill_len_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.int32)
        self.x_len_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.int32)
        self.prompt_len_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.int32)
        self.max_step_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.int32)
        self.completed_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.bool)
        self.y_buf = torch.full(
            (self.max_batch_size, decoder.max_seq_length), decoder.EOS, device=device, dtype=torch.int32
        )

        self.top_k_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.int64)
        self.top_p_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.float32)
        self.temperature_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.float32)
        self.repetition_penalty_buf = torch.zeros(self.max_batch_size, device=device, dtype=torch.float32)

        self.extra_buffers = extra_buffer_factory(self.max_batch_size, decoder)

        self.pool: Optional[torch.cuda._POOL_HANDLE] = None

        target_batch_sizes = self._target_batch_sizes()
        self.target_bsz = sorted(set(target_batch_sizes))
        self.graph_cache: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_error = None
        if self.graph_applicable:
            try:
                for bsz in self.target_bsz:
                    self.graph_cache[bsz] = self._capture_graph(bsz)
            except (RuntimeError, OSError, NotImplementedError) as exc:
                self.graph_cache.clear()
                self.pool = None
                self.graph_applicable = False
                self.graph_enabled = False
                self.graph_error = exc
                warnings.warn(f"CUDA Graph capture failed; disabling Graph: {exc}", RuntimeWarning)

    def _should_use_graph(self) -> bool:
        return bool(
            self.graph_enabled
            and self.graph_applicable
            and torch.cuda.is_available()
            and torch.version.cuda is not None
            and os.environ.get("CUDAGraph", "1") != "0"
        )

    def _target_batch_sizes(self) -> list[int]:
        sizes = {1, self.max_batch_size}
        if self.max_batch_size >= 8:
            sizes.update([2, 4, 8])
        if self.max_batch_size > 8:
            sizes.update(range(16, self.max_batch_size + 1, 8))
        if self.max_batch_size >= 2:
            sizes.add(min(2, self.max_batch_size))
        if self.max_batch_size >= 4:
            sizes.add(min(4, self.max_batch_size))
        return sorted(sizes)

    def _select_bsz(self, requested: int) -> int:
        for bsz in self.target_bsz:
            if requested <= bsz:
                return bsz
        return self.target_bsz[-1]

    def _capture_graph(self, bsz: int):
        with torch.cuda.device(self.device):
            args, kwds = self.decoder.graph_capture_inputs(self, bsz)
            args = _slice_tensors(args, bsz)
            kwds = _slice_tensors(kwds, bsz)
            kwds["pool"] = self.pool
            graph = self.decoder.capture(*args, **kwds)
            if self.pool is None:
                self.pool = graph.pool()
            return graph

    def _acquire_graph(self, bsz: int) -> torch.cuda.CUDAGraph:
        if not self.graph_applicable:
            raise RuntimeError("CUDAGraph is not applicable for this decoder/device")
        if bsz > self.max_batch_size:
            raise ValueError(f"Requested batch size {bsz} exceeds max_batch_size {self.max_batch_size}")
        target = self._select_bsz(bsz)
        return self.graph_cache[target]

    def prefill(self, request: T2SRequest) -> T2SSession:
        if len(request.x) > self.decoder.max_batch_size:
            raise ValueError(
                f"Request batch size {len(request.x)} exceeds decoder max batch size {self.decoder.max_batch_size}"
            )

        session = T2SSession(
            self.decoder,
            request,
            device=self.device,
            dtype=self.dtype,
        )

        kv_cache = self.decoder.init_cache(session.bsz)

        torch_profiler = TorchProfiler(request.debug)
        with torch_profiler.profiler(), timer("Torch.Prefill", debug=request.debug), torch.inference_mode():
            torch_profiler.start()
            xy_pos = self.decoder.embed(session.x, request.prompts, session.bert_feature)
            xy_dec = self.decoder.h.prefill(xy_pos, kv_cache, session.attn_mask)
            gather_index = (session.input_pos - 1).long().view(session.bsz, 1, 1)
            session.prefill_hidden = torch.take_along_dim(xy_dec, gather_index, dim=1)
            torch_profiler.end()

        session.kv_cache = kv_cache

        session.start_time = time.perf_counter()
        session.total_tokens = 0
        return session

    def free_slots(self) -> int:
        return sum(1 for s in self.slot_sessions if s is None)

    def _bind_waiting_sessions(self) -> None:
        while self.waiting_bind and self.free_slots() >= self.waiting_bind[0].bsz:
            session = self.waiting_bind[0]
            if self.bind_session(session):
                self.waiting_bind.popleft()
            else:
                break

    def _assign_slots(self, bsz: int) -> Optional[list[int]]:
        indices = [i for i, s in enumerate(self.slot_sessions) if s is None]
        if len(indices) < bsz:
            return None
        return indices[:bsz]

    def bind_session(self, session: T2SSession) -> bool:
        slots = self._assign_slots(session.bsz)
        if slots is None:
            return False

        for slot_idx in slots:
            self.slot_sessions[slot_idx] = session

        idx_tensor = torch.tensor(slots, device=self.device, dtype=torch.long)
        take = slice(0, len(slots))

        # Input pos and step
        self.input_pos_buf.index_copy_(0, idx_tensor, session.input_pos[take])
        self.step_buf.index_fill_(0, idx_tensor, 0)

        # Prefill hidden goes straight into xy_dec buffer
        self.xy_dec_buf.index_copy_(0, idx_tensor, session.prefill_hidden[take])

        # KV Cache
        for (k_buf, v_buf), (k_src, v_src) in zip(self.kv_cache_buf, session.kv_cache):
            k_buf.index_copy_(0, idx_tensor, k_src[take])
            v_buf.index_copy_(0, idx_tensor, v_src[take])

        # Sampling params
        self.top_k_buf.index_fill_(0, idx_tensor, session.request.top_k)
        self.top_p_buf.index_fill_(0, idx_tensor, session.request.top_p)
        self.temperature_buf.index_fill_(0, idx_tensor, session.request.temperature)
        self.repetition_penalty_buf.index_fill_(0, idx_tensor, session.request.repetition_penalty)
        self.prefill_len_buf.index_copy_(0, idx_tensor, session.prefill_len[take])
        self.x_len_buf.index_copy_(0, idx_tensor, session.x_lens[take])
        self.prompt_len_buf.index_fill_(0, idx_tensor, session.prompt_len)
        max_steps = (
            session.max_decode_steps
            if session.request.early_stop_num == -1
            else min(session.max_decode_steps, session.request.early_stop_num)
        )
        self.max_step_buf.index_fill_(0, idx_tensor, max_steps)
        self.completed_buf.index_fill_(0, idx_tensor, False)

        # Token buffer
        self.y_buf.index_fill_(0, idx_tensor, self.decoder.EOS)
        for i, slot_idx in enumerate(slots):
            self.y_buf[slot_idx, : session.prompt_len] = session.y[i, : session.prompt_len]

        self.session_slots[session.id] = slots
        session.slot_indices = slots
        self.decoder.bind_session_hook(session, self)
        self.bound_sessions.append(session)
        return True

    def unbind_session(self, session: T2SSession) -> None:
        slots = self.session_slots.pop(session.id, [])
        for slot_idx in slots:
            self.slot_sessions[slot_idx] = None
        self.input_pos_buf[slots] = 1
        self.step_buf[slots] = 0
        self.completed_buf[slots] = False
        self.decoder.unbind_session_hook(session, self)
        if session in self.bound_sessions:
            self.bound_sessions.remove(session)

    def decode_step(self) -> list[T2SSession]:
        if not self.bound_sessions and not self.waiting_bind:
            return []

        decoder = self.decoder
        finished_sessions: list[T2SSession] = []

        forward_slots: list[int] = []
        prefill_slots: list[int] = []
        for slot_idx, session in enumerate(self.slot_sessions):
            if session is None or self.completed_buf[slot_idx].item():
                continue
            if self.step_buf[slot_idx].item() != 0:
                forward_slots.append(slot_idx)

        use_graph = self._should_use_graph() and bool(forward_slots)

        debug = any(session.request.debug for session in self.bound_sessions)

        try:
            with torch.inference_mode():
                if forward_slots:
                    forward_len = max(forward_slots) + 1
                    model_len = self._select_bsz(forward_len) if use_graph else forward_len
                    kwds = decoder.pre_forward_slots_hook(list(range(model_len)), self)
                    if use_graph:
                        graph = self._acquire_graph(model_len)
                        graph.replay()
                    else:
                        self.xy_dec_buf[:forward_len] = decoder.h(
                            self.xy_pos_buf[:forward_len],
                            self.input_pos_buf[:forward_len],
                            _slice_tensors(self.kv_cache_buf, forward_len),
                            **kwds,
                        )

                # Bind any newly ready sessions after decode and before sampling
                if self.waiting_bind and self.free_slots() > 0:
                    self._bind_waiting_sessions()
                    if not debug:
                        debug = any(session.request.debug for session in self.bound_sessions)

                for slot_idx, session in enumerate(self.slot_sessions):
                    if session is None:
                        continue
                    if session.step_count == 0:
                        prefill_slots.append(slot_idx)

                active_slots = forward_slots + prefill_slots
                if not active_slots:
                    return []

                idx_tensor = torch.tensor(active_slots, device=self.device, dtype=torch.long)

                logits_all = decoder.ar_predict_layer(self.xy_dec_buf[idx_tensor].squeeze(1))

                first_step_mask = (self.step_buf[idx_tensor] == 0).bool()

                if torch.any(first_step_mask):
                    logits_all[first_step_mask, -1] = float("-inf")

                sampler = self.sampler

                samples = sampler(
                    logits=logits_all,
                    valid_lens=self.prompt_len_buf[idx_tensor] + self.step_buf[idx_tensor],
                    previous_tokens=self.y_buf[idx_tensor],
                    repetition_penalty=self.repetition_penalty_buf[idx_tensor],
                    temperature=self.temperature_buf[idx_tensor],
                    top_k=self.top_k_buf[idx_tensor],
                    top_p=self.top_p_buf[idx_tensor],
                )

                argmax_token = torch.argmax(logits_all, dim=-1)
                eos_mask_all = torch.logical_or((argmax_token == decoder.EOS), (samples.squeeze(1) == decoder.EOS))

                y_emb_all = decoder.ar_audio_embedding(samples.clamp_max_(decoder.vocab_size - 2))
                pos_offset = self.step_buf[idx_tensor] + self.prompt_len_buf[idx_tensor] + 1
                next_xy_all = decoder.ar_audio_position(pos_offset, y_emb_all)

                sample_vals = samples.squeeze(1)
                write_pos = self.prompt_len_buf[idx_tensor] + self.step_buf[idx_tensor]

                # Only write for slots that are not already done and not EOS
                write_mask = (~self.completed_buf[idx_tensor]) & (~eos_mask_all)
                if torch.any(write_mask):
                    write_slots = idx_tensor[write_mask]
                    self.y_buf[write_slots, write_pos[write_mask]] = sample_vals[write_mask]
                    self.input_pos_buf[write_slots] += 1

                new_step = self.step_buf[idx_tensor].clone()
                new_step[write_mask] += 1
                forced_stop = new_step >= self.max_step_buf[idx_tensor]
                self.step_buf[idx_tensor] = new_step
                self.completed_buf[idx_tensor] |= eos_mask_all | forced_stop
                done_slots = idx_tensor[self.completed_buf[idx_tensor]]
                # Graph buckets also execute inactive rows, so their cache position must remain valid.
                self.input_pos_buf[done_slots] = 1

                # Update next xy for continuing slots
                continue_mask = ~(self.completed_buf[idx_tensor])
                if torch.any(continue_mask):
                    cont_slots = idx_tensor[continue_mask]
                    cont_xy = next_xy_all[continue_mask]
                    self.xy_pos_buf[cont_slots] = cont_xy

                # Sync back
                for slots in list(self.session_slots.values()):
                    if not slots:
                        continue
                    session = self.slot_sessions[slots[0]]
                    if session is None:
                        continue
                    session.output_lengths = self.step_buf[slots].tolist()
                    session.step_count = max(session.output_lengths)
                    session.total_tokens = sum(session.output_lengths)
                    session.completed = self.completed_buf[slots].clone()
                    if bool(torch.all(session.completed)):
                        session.y_results = [
                            self.y_buf[slot, session.prompt_len : session.prompt_len + length].clone()
                            for slot, length in zip(slots, session.output_lengths)
                        ]
                        finished_sessions.append(session)
                        self.unbind_session(session)

                decoder.post_forward_slots_hook(forward_slots, self)

        finally:
            if debug:
                _synchronize_device(self.device)

        return finished_sessions
