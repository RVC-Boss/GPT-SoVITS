from __future__ import annotations

import mlx.core as mx

from ..structs_mlx import KVCache
from ..t2s_model_abc import (
    AttentionABC,
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

    def __call__(self, x: Array, input_pos: Array, max_idx: int, kv_cache: KVCache, cache_idx: Array, attn_mask: Array):
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj(x).split(3, axis=-1)

        q, k, v = map(lambda x: x.reshape(bsz, seqlen, self.n_head, self.head_dim), (q, k, v))

        q, k, v = map(lambda x: x.swapaxes(1, 2), (q, k, v))

        kv_cache = self.kc_class.update_cache(input_pos, k, v, kv_cache, cache_idx)

        q, k, v = map(lambda x: x[..., :max_idx, :], (q, *kv_cache))

        mask = attn_mask[..., :max_idx]

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        attn = attn.swapaxes(1, 2).reshape(bsz, seqlen, -1)

        attn = self.out_proj(attn)

        return attn


class TransformerBlock(TransformerBlockABC):
    def __init__(self, n_head: int, ffn_dim: int, hidden_dim: int, max_seq_length: int) -> None:
        super().__init__(n_head, ffn_dim, hidden_dim, max_seq_length)

        self.attention = Attention(n_head, hidden_dim, max_seq_length)


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
    ) -> None:
        super().__init__(
            hidden_dim,
            n_layer,
            n_head,
            ffn_dim,
            vocab_size,
            max_seq_length,
            max_batch_size,
        )

        self.layers = [
            TransformerBlock(
                n_head,
                ffn_dim,
                hidden_dim,
                max_seq_length,
            )
            for _ in range(n_layer)
        ]


class T2SDecoder(T2SDecoderABC):
    def __init__(
        self,
        config: dict,
        max_seq_length: int = 1500,
        max_batch_size: int = 10,
    ) -> None:
        super().__init__(config, max_seq_length, max_batch_size)

        self.h = TransformerDecoder(
            self.hidden_dim, self.n_layer, self.n_head, self.ffn_dim, self.vocab_size, max_seq_length, max_batch_size
        )

        self.kv_class = KVCacheHND
