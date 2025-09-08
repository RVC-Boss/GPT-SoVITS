"""
Modified From https://github.com/XXXXRT666/GPT-SoVITS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, MutableSequence, Protocol, TypeAlias, cast

import mlx.core as mx
import torch

from ..PyTorch.structs import T2SRequest
from .sample_funcs_mlx import SampleProtocolMLX, sample_naive

Tensor = torch.Tensor
Array = mx.array


@dataclass(slots=True)
class T2SRequestMLX:
    x: List[Array]
    x_lens: Array
    prompts: Array
    bert_feature: List[Array]
    valid_length: int
    top_k: int = 5
    top_p: float = 1
    early_stop_num: int = -1
    temperature: float = 1.0
    repetition_penalty: float = 1.35

    @classmethod
    def from_torch(cls, request: T2SRequest) -> T2SRequestMLX:
        x = list(map(lambda tensor: mx.array(tensor.cpu()), request.x))
        x_lens = mx.array(request.x_lens.cpu())
        prompts = mx.array(request.prompts.cpu())
        bert_feature = list(map(lambda tensor: mx.array(tensor.cpu()), request.bert_feature))

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
        )


KVCache: TypeAlias = tuple[Array, Array]
KVCacheQ: TypeAlias = tuple[tuple[Array, Array, Array], tuple[Array, Array, Array], tuple[int, int]]


class KVCacheProtocol(Protocol):
    @staticmethod
    def empty(kv_cache: KVCache | KVCacheQ) -> None: ...

    @staticmethod
    def update_cache(
        input_pos: Array, k_val: Array, v_val: Array, kv_cache: KVCache | KVCacheQ, cache_idx: Array
    ) -> KVCache | KVCacheQ: ...

    @staticmethod
    def prefill_kv(k_val: Array, v_val: Array, kv_cache: KVCache | KVCacheQ) -> None: ...

    @staticmethod
    def init_cache(
        batch_size: int, max_seq_length: int, n_heads: int, head_dim: int, dtype: mx.Dtype, *args, **kwds
    ) -> KVCache | KVCacheQ: ...


class T2SDecoderProtocol(Protocol):
    max_seq_length: int
    EOS: int
    n_head: int

    def embed(self, x: list[Array], y: Array, bert_features: list[Array]) -> Array: ...


class T2SSessionMLX:
    def __init__(
        self,
        decoder: T2SDecoderProtocol,
        request_torch: T2SRequest,
        sample_func: type[SampleProtocolMLX] = sample_naive,
        device: mx.Device = mx.Device(mx.cpu),
        dtype: mx.Dtype = mx.float32,
    ):
        with mx.stream(device):
            request = T2SRequestMLX.from_torch(request_torch)

            self.decoder = decoder
            self.request = request
            self.device = device
            self.dtype = dtype

            bsz = len(request.x)
            y_len: int = cast(tuple[int, ...], request.prompts.shape)[-1]
            self.bsz = bsz
            self.y_len = y_len

            # Cache
            self.kv_cache: MutableSequence[KVCache | KVCacheQ]
            self.sample = sample_func()

            # Forward args
            self.x = [i.astype(mx.int32) for i in request.x]
            self.x_lens = request.x_lens.astype(mx.int32)
            self.y = mx.zeros((bsz, decoder.max_seq_length)).astype(mx.int32)
            self.y[:, : cast(tuple[int, ...], request.prompts.shape)[-1]] = request.prompts.astype(mx.int32)
            self.bert_feature = [i.astype(dtype) for i in request.bert_feature]

            self.prefill_len = self.x_lens + cast(tuple[int, ...], request.prompts.shape)[1]

            self.input_pos = mx.zeros_like(self.prefill_len)
            self.input_pos += self.prefill_len

            # EOS
            self.completed = mx.array([False] * len(self.x)).astype(mx.bool_)
            self.y_results: List[Array] = [None] * len(self.x)  # type: ignore

            self.xy_pos = decoder.embed(self.x, request.prompts, self.bert_feature)

            max_len = int(self.prefill_len.max(-1))
            attn_mask = mx.zeros(shape=(bsz, max_len, max_len), dtype=mx.bool_)

            for bs in range(bsz):
                pos = int(self.x_lens[bs])
                seq_len = pos + y_len

                attn_mask[bs, :seq_len, :pos] = True

                ar_mask = ~mx.triu(
                    x=mx.ones(
                        shape=(
                            y_len,
                            y_len,
                        ),
                        dtype=mx.bool_,
                    ),
                    k=1,
                )
                attn_mask[bs, pos:seq_len, pos:seq_len] = ar_mask

            attn_mask = mx.repeat(mx.expand_dims(attn_mask, 1), decoder.n_head, 1)
            self.attn_mask = attn_mask

            mx.eval(self.attn_mask)
