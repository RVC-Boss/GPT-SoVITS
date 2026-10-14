from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import MutableSequence
from typing import Optional, TypeVar

import mlx.core as mx
import torch

from ...logger import timer

from .sample_funcs_mlx import sample_naive_mlx as sample_naive
from .structs_mlx import KVCache, T2SRequest, T2SSessionMLX as T2SSession
from .t2s_model_abc_mlx import T2SDecoderABC


T = TypeVar("T")
Array = mx.array


def _slice_arrays(obj: T, bsz: int) -> T:
    if isinstance(obj, Array):
        if obj.ndim > 0 and obj.shape[0] >= bsz:
            return obj[:bsz]  # type: ignore
        return obj  # type: ignore
    if isinstance(obj, list):
        return [_slice_arrays(o, bsz) for o in obj]  # type: ignore
    if isinstance(obj, tuple):
        return tuple(_slice_arrays(o, bsz) for o in obj)  # type: ignore
    if isinstance(obj, dict):
        return {k: _slice_arrays(v, bsz) for k, v in obj.items()}  # type: ignore
    return obj


def _synchronize_device(device: torch.device) -> None:
    device_type = device.type
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "xpu":
        torch.xpu.synchronize()
    elif device_type == "mtia":
        torch.mtia.synchronize()
    elif device_type == "cpu":
        pass


class ModelRunner:
    """Model-side executor handling prefill, decode"""

    def __init__(
        self,
        decoder: T2SDecoderABC,
        device: mx.Device,
        dtype: mx.Dtype,
        waiting_bind: deque[T2SSession],
    ):
        self.max_batch_size = decoder.max_batch_size

        self.decoder = decoder
        self.device = device
        self.dtype = dtype

        self.waiting_bind: deque[T2SSession] = waiting_bind
        self.slot_sessions: list[Optional[T2SSession]] = [None] * decoder.max_batch_size
        self.session_slots: dict[int, list[int]] = {}
        self.bound_sessions: list[T2SSession] = []
        self.sampler = sample_naive()

        extra_buffer_factory = decoder.extra_buffer_factory
        is_applicable = decoder.graph_applicable

        cuda_ok = (
            torch.cuda.is_available() and torch.version.cuda is not None and os.environ.get("CUDAGraph", "1") != "0"
        )
        self.graph_applicable = bool(cuda_ok and is_applicable)

        dtype_bias = decoder.bert_proj.bias.dtype

        # Shared runtime buffers for both eager and graph replay
        self.xy_pos_buf = mx.zeros((self.max_batch_size, 1, decoder.embedding_dim), dtype=dtype_bias)
        self.xy_dec_buf = mx.zeros_like(self.xy_pos_buf)
        self.input_pos_buf = mx.zeros((self.max_batch_size,), dtype=mx.int32)
        self.kv_cache_buf: MutableSequence[KVCache] = decoder.init_cache(self.max_batch_size)

        self.step_buf = mx.zeros((self.max_batch_size), dtype=mx.int32)
        self.prefill_len_buf = mx.zeros((self.max_batch_size), dtype=mx.int32)
        self.x_len_buf = mx.zeros((self.max_batch_size), dtype=mx.int32)
        self.prompt_len_buf = mx.zeros((self.max_batch_size), dtype=mx.int32)
        self.max_step_buf = mx.zeros((self.max_batch_size), dtype=mx.int32)
        self.completed_buf = mx.zeros((self.max_batch_size), dtype=mx.bool_)
        self.y_buf = mx.full((self.max_batch_size, decoder.max_seq_length), decoder.EOS, dtype=mx.int32)

        self.top_k_buf = mx.zeros(self.max_batch_size, dtype=mx.int32)
        self.top_p_buf = mx.zeros(self.max_batch_size, dtype=mx.float32)
        self.temperature_buf = mx.zeros(self.max_batch_size, dtype=mx.float32)
        self.repetition_penalty_buf = mx.zeros(self.max_batch_size, dtype=mx.float32)

        self.extra_buffers = extra_buffer_factory(self.max_batch_size, decoder)

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

        with timer("MLX.Prefill", debug=request.debug):
            xy_pos = self.decoder.embed(session.x, session.request.prompts, session.bert_feature)
            xy_dec = self.decoder.h.prefill(xy_pos, kv_cache, session.attn_mask)
            session.prefill_hidden = mx.take_along_axis(xy_dec, session.input_pos - 1, axis=1)

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

        idx_array = mx.array(slots, dtype=mx.int64)
        take = slice(0, len(slots))

        # Input pos and step
        mx.put_along_axis(self.input_pos_buf, idx_array, session.input_pos[take], axis=0)
        self.step_buf[idx_array] = 0

        # Prefill hidden goes straight into xy_dec buffer
        mx.put_along_axis(self.xy_dec_buf, idx_array, session.prefill_hidden[take], axis=0)

        # KV Cache
        for (k_buf, v_buf), (k_src, v_src) in zip(self.kv_cache_buf, session.kv_cache):
            mx.put_along_axis(k_buf, idx_array, k_src[take], axis=0)
            mx.put_along_axis(v_buf, idx_array, v_src[take], axis=0)

        # Sampling params
        self.top_k_buf[idx_array] = session.request.top_k
        self.top_p_buf[idx_array] = session.request.top_p
        self.temperature_buf[idx_array] = session.request.temperature
        self.repetition_penalty_buf[idx_array] = session.request.repetition_penalty
        mx.put_along_axis(self.prefill_len_buf, idx_array, session.prefill_len[take], axis=0)
        mx.put_along_axis(self.x_len_buf, idx_array, session.x_lens[take], axis=0)
        self.prompt_len_buf[idx_array] = session.prompt_len
        max_steps = (
            session.max_decode_steps
            if session.request.early_stop_num == -1
            else min(session.max_decode_steps, session.request.early_stop_num)
        )
        self.max_step_buf[idx_array] = max_steps
        self.completed_buf[idx_array] = False

        # Token buffer
        self.y_buf[idx_array] = self.decoder.EOS
        self.y_buf[idx_array, : session.prompt_len] = session.y

        self.session_slots[session.id] = slots
        session.slot_indices = slots
        self.decoder.bind_session_hook(session, self)
        self.bound_sessions.append(session)

        return True

    def unbind_session(self, session: T2SSession) -> None:
        slots = self.session_slots.pop(session.id, [])
        for slot_idx in slots:
            self.slot_sessions[slot_idx] = None
        self.input_pos_buf[slots] = 0
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
            if session is None:
                continue
            if self.step_buf[slot_idx].item() != 0:
                forward_slots.append(slot_idx)

        debug = any(session.request.debug for session in self.bound_sessions)

        try:
            if forward_slots:
                kwds = decoder.pre_forward_slots_hook(forward_slots, self)
                forward_len = max(forward_slots) + 1
                self.xy_dec_buf[:forward_len] = decoder.h(
                    self.xy_pos_buf[:forward_len],
                    self.input_pos_buf[:forward_len],
                    _slice_arrays(self.kv_cache_buf, forward_len),
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
            idx_array = mx.array(active_slots, dtype=mx.int32)
            if not active_slots:
                return []

            logits_all = decoder.ar_predict_layer(self.xy_dec_buf[idx_array].squeeze(1))

            first_step_mask = mx.equal(self.step_buf[idx_array], 0).astype(mx.bool_)

            if mx.any(first_step_mask):
                logits_all[first_step_mask, -1] = float("-inf")

            samples: Array = self.sampler(
                logits=logits_all,
                valid_lens=self.input_pos_buf[idx_array] + self.prefill_len_buf[idx_array],
                previous_tokens=self.y_buf[idx_array],
                repetition_penalty=self.repetition_penalty_buf[idx_array],
                temperature=self.temperature_buf[idx_array],
                top_k=self.top_k_buf[idx_array],
                top_p=self.top_p_buf[idx_array],
            )

            argmax_token = mx.argmax(logits_all, axis=-1)
            eos_mask_all = mx.logical_or(mx.equal(argmax_token, decoder.EOS), mx.equal(samples.squeeze(1), decoder.EOS))

            samples_clamp = samples[:]
            samples_clamp[samples_clamp >= decoder.vocab_size - 1] = decoder.vocab_size - 2

            y_emb_all = decoder.ar_audio_embedding(samples_clamp)
            pos_offset = self.step_buf[idx_array] + self.prefill_len_buf[idx_array]
            next_xy_all = decoder.ar_audio_position(pos_offset, y_emb_all)

            sample_vals = samples.squeeze(1)
            write_pos = self.prompt_len_buf[idx_array] + self.step_buf[idx_array]

            # Only write for slots that are not already done and not EOS
            write_mask = (~self.completed_buf[idx_array]) & (~eos_mask_all)
            if mx.any(write_mask):
                write_slots = idx_array[write_mask]
                self.y_buf[write_slots, write_pos[write_mask]] = sample_vals[write_mask]
                self.input_pos_buf[write_slots] += 1

                new_step = self.step_buf[idx_array]
                new_step[write_mask] += 1
                forced_stop = new_step >= self.max_step_buf[idx_array]
                self.step_buf[idx_array] = new_step

                self.completed_buf[idx_array] |= eos_mask_all | forced_stop

            # Update next xy for continuing slots
            continue_mask = ~(self.completed_buf[idx_array])
            if mx.any(continue_mask):
                cont_slots = idx_array[continue_mask]
                cont_xy = next_xy_all[continue_mask]
                self.xy_pos_buf[cont_slots] = cont_xy

            # Sync back
            for slots in list(self.session_slots.values()):
                if not slots:
                    continue
                session = self.slot_sessions[slots[0]]
                if session is None:
                    continue
                session_step = int(self.step_buf[slots[0]].item())
                session.step_count = session_step
                session.total_tokens = int(session_step * session.bsz)
                session.completed = self.completed_buf[slots]
                if bool(mx.all(session.completed)):
                    finished_sessions.append(session)
                    self.unbind_session(session)

            decoder.post_forward_slots_hook(forward_slots, self)

        finally:
            pass

        return finished_sessions
