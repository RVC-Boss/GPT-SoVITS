"""
Modified From https://github.com/XXXXRT666/GPT-SoVITS
"""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import Optional, Protocol, Union

import mlx.core as mx
from rich.progress import Progress, TaskID

from ...PyTorch.AR.structs import T2SEngineProtocol, T2SRequest, T2SRequestHandle, T2SResult, T2SStreamResponse


Array = mx.array


@dataclass
class T2SRequestMLX:
    x: list[Array]
    x_lens: Array
    prompts: Array
    bert_feature: list[Array]
    valid_length: int
    top_k: int = 5
    top_p: float = 1
    early_stop_num: int = -1
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    request_id: Union[str, int] = -1
    stream_interval: Optional[int] = None
    return_partial: bool = True
    debug: bool = False

    @classmethod
    def from_torch(cls, request: T2SRequest) -> T2SRequestMLX:
        x = list(map(lambda tensor: mx.array(tensor.cpu()), request.x))  # type: ignore
        x_lens = mx.array(request.x_lens.cpu())  # type: ignore
        prompts = mx.array(request.prompts.cpu())  # type: ignore
        bert_feature = list(map(lambda tensor: mx.array(tensor.cpu()), request.bert_feature))  # type: ignore

        return cls(
            x,
            x_lens,
            prompts,
            bert_feature,
            request.valid_length,
            request.top_k,
            request.top_p,
            request.early_stop_num,
            request.temperature,
            request.repetition_penalty,
            request.debug,
        )


KVCache = tuple[Array, ...]


class KVCacheProtocol(Protocol):
    @staticmethod
    def empty(kv_cache: KVCache) -> None: ...

    @staticmethod
    def update_cache(input_pos: Array, k_val: Array, v_val: Array, kv_cache: KVCache) -> KVCache: ...

    @staticmethod
    def prefill_kv(k_val: Array, v_val: Array, kv_cache: KVCache) -> None: ...

    @staticmethod
    def init_cache(batch_size: int, max_seq_length: int, n_heads: int, head_dim: int, dtype: mx.Dtype) -> KVCache: ...


class T2SDecoderProtocol(Protocol):
    max_seq_length: int
    EOS: int
    n_head: int

    def embed(self, x: list[Array], y: Array, bert_features: list[Array]) -> Array: ...


cpu = mx.Device(mx.cpu)


class T2SSessionMLX:
    def __init__(
        self,
        decoder: T2SDecoderProtocol,
        request_torch: T2SRequest,
        device: mx.Device = cpu,
        dtype: mx.Dtype = mx.float32,
    ):
        with mx.stream(device):
            request = T2SRequestMLX.from_torch(request_torch)

            self.decoder = decoder
            self.request = request
            self.device = device
            self.dtype = dtype

            bsz = len(request.x)
            prompt_len = request.prompts.shape[-1]
            self.bsz = bsz

            self.step_count = 0
            self.request_id = request.request_id or id(self)
            self.stream_interval = request.stream_interval if request.stream_interval is not None else 0
            self.last_stream_step = 0

            # Cache in prefill
            self.kv_cache: MutableSequence[KVCache]

            # Forward args
            self.x = [i.astype(mx.int32) for i in request.x]
            self.x_lens = request.x_lens.astype(mx.int32)
            self.y = request.prompts.astype(mx.int32)
            self.prompt_len = prompt_len
            self.bert_feature = [i.astype(dtype) for i in request.bert_feature]

            self.prefill_len = self.x_lens + request.prompts.shape[1]

            self.input_pos = mx.zeros_like(self.prefill_len)
            self.input_pos += self.prefill_len
            if bsz == 1:
                self.input_pos = self.input_pos.squeeze(0)  # 30% Performance Improvement in bsz=1

            self.max_decode_steps = min(int(decoder.max_seq_length - int(self.input_pos.max().item())), 640)
            self.max_decode_steps = max(1, self.max_decode_steps)

            # EOS
            self.completed = mx.array([False] * len(self.x)).astype(mx.bool_)
            self.y_results: list[Array] = [None] * len(self.x)  # type: ignore

            max_len = int(self.prefill_len.max(-1))
            attn_mask = mx.zeros(shape=(bsz, max_len, max_len), dtype=mx.bool_)

            for bs in range(bsz):
                pos = int(self.x_lens[bs])
                seq_len = pos + prompt_len

                attn_mask[bs, :seq_len, :pos] = True

                ar_mask = ~mx.triu(
                    x=mx.ones(
                        shape=(
                            prompt_len,
                            prompt_len,
                        ),
                        dtype=mx.bool_,
                    ),
                    k=1,
                )
                attn_mask[bs, pos:seq_len, pos:seq_len] = ar_mask

            attn_mask = mx.expand_dims(attn_mask, 1)
            self.attn_mask = attn_mask

            self.prefill_hidden: Array

            self.start_time: float = 0.0
            self.total_tokens: int = 0

            self.progress_task: Optional[TaskID] = None
            self.progress: Optional[Progress] = None

            self.id: int = id(self)
            self.slot_indices: list[int] = []


__all__ = [
    "T2SRequestMLX",
    "T2SSessionMLX",
    "KVCache",
    "KVCacheProtocol",
    "T2SDecoderProtocol",
    "T2SEngineProtocol",
    "T2SRequest",
    "T2SRequestHandle",
    "T2SResult",
    "T2SStreamResponse",
]
