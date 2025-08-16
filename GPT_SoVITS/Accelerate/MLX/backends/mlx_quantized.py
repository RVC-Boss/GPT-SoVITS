from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..structs_mlx import KVCacheQ
from ..t2s_model_abc import (
    AttentionABC,
    KVCache,
    KVCacheHND,
    T2SDecoderABC,
    TransformerBlockABC,
    TransformerDecoderABC,
)

Array = mx.array


class Attention(AttentionABC):
    def __init__(self, n_head: int, hidden_dim: int, max_seq_length: int):
        super().__init__(n_head, hidden_dim, max_seq_length)
        self.kc_class = KVCacheHND

    @staticmethod
    def quantized_scaled_dot_product_attention(
        queries: Array,
        q_keys: tuple[Array, Array, Array],
        q_values: tuple[Array, Array, Array],
        scale: float,
        mask: Array,
        group_size: int = 32,
        bits: int = 8,
    ) -> Array:
        queries *= scale

        scores = mx.quantized_matmul(queries, *q_keys, transpose=True, group_size=group_size, bits=bits)
        scores = mx.where(mask, scores, -mx.inf)
        scores = mx.softmax(scores, axis=-1, precise=True)  # type: ignore
        out = mx.quantized_matmul(scores, *q_values, transpose=False, group_size=group_size, bits=bits)

        return out

    def __call__(self, x: Array, input_pos: Array, kv_cache: KVCache | KVCacheQ, cache_idx: Array, attn_mask: Array):
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj(x).split(3, axis=-1)

        q, k, v = map(lambda x: x.reshape(bsz, seqlen, self.n_head, self.head_dim), (q, k, v))

        q, k, v = map(lambda x: x.swapaxes(1, 2), (q, k, v))

        kv_cache = self.kc_class.update_cache(input_pos, k, v, kv_cache, cache_idx)
        assert len(kv_cache) == 2

        max_idx = int(input_pos.max())

        q, k, v = map(lambda x: x[..., :max_idx, :], (q, *kv_cache))

        mask = attn_mask[..., :max_idx]

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        attn = attn.swapaxes(1, 2).reshape(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj(attn)

        return attn

    # def __call__(self, x: Array, input_pos: Array, kv_cache: KVCache | KVCacheQ, cache_idx: Array, attn_mask: Array):
    #     bsz, seqlen, _ = x.shape

    #     q, k, v = self.in_proj(x).split(3, axis=-1)

    #     q, k, v = map(lambda x: x.reshape(bsz, seqlen, self.n_head, self.head_dim), (q, k, v))

    #     q, k, v = map(lambda x: x.swapaxes(1, 2), (q, k, v))

    #     kv_cache = self.kc_class.update_cache(input_pos, k, v, kv_cache, cache_idx)

    #     assert len(kv_cache) == 3
    #     (k_q, k_s, k_b), (v_q, v_s, v_b), (group_size, bits) = kv_cache

    #     k_q, k_s, k_b, v_q, v_s, v_b = map(lambda x: x[..., : int(input_pos.max()), :], (k_q, k_s, k_b, v_q, v_s, v_b))

    #     mask = attn_mask[..., : int(input_pos.max())]

    #     attn = Attention.quantized_scaled_dot_product_attention(
    #         q,
    #         (k_q, k_s, k_b),
    #         (v_q, v_s, v_b),
    #         self.scale,
    #         mask,
    #         group_size,
    #         bits,
    #     )

    #     attn = attn.swapaxes(1, 2).reshape(bsz, seqlen, self.hidden_dim)

    #     output = self.out_proj(attn)

    #     return output


class TransformerBlock(TransformerBlockABC):
    def __init__(self, n_head: int, ffn_dim: int, hidden_dim: int, max_seq_length: int, *args, **kwds) -> None:
        super().__init__(n_head, ffn_dim, hidden_dim, max_seq_length, *args, **kwds)

        self.attention = Attention(n_head, hidden_dim, max_seq_length, *args, **kwds)


class TransformerDecoder(TransformerDecoderABC):
    def __init__(
        self,
        hidden_dim: int,
        n_layer: int,
        n_head: int,
        ffn_dim: int,
        vocab_size: int,
        max_seq_length: int,
        max_batch_size: int,
        *args,
        **kwds,
    ) -> None:
        super().__init__(
            hidden_dim,
            n_layer,
            n_head,
            ffn_dim,
            vocab_size,
            max_seq_length,
            max_batch_size,
            *args,
            **kwds,
        )

        self.layers = [
            TransformerBlock(
                n_head,
                ffn_dim,
                hidden_dim,
                max_seq_length,
                *args,
                **kwds,
            )
            for _ in range(n_layer)
        ]


class T2SDecoder(T2SDecoderABC):
    def __init__(
        self,
        config: dict,
        max_seq_length: int = 2000,
        max_batch_size: int = 10,
    ) -> None:
        super().__init__(config, max_seq_length, max_batch_size)

        self.h = TransformerDecoder(
            self.hidden_dim, self.n_layer, self.n_head, self.ffn_dim, self.vocab_size, max_seq_length, max_batch_size
        )

        self.kv_class = KVCacheHND
        self.group_size = 32
        self.bits = 8
        self.mode = "affine"

    def set_mode(self, mode: str):
        assert mode in ["affine", "mxfp4"]
        self.mode = mode
        if self.mode == "mxfp4":
            self.bits = 4
        else:
            self.bits = 8

    def quantized(self):
        nn.quantize(self, self.group_size, self.bits, mode=self.mode)
        # for layer in self.h.layers:
        #     nn.quantize(layer.feed_forward, self.group_size, self.bits)
        #     nn.quantize(layer.attention, self.group_size, self.bits)
