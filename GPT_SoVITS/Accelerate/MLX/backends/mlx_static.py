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

    def __call__(self, x: Array, input_pos: Array, kv_cache: KVCache, cache_idx: Array, attn_mask: Array):
        bsz, seqlen, _ = x.shape

        qkv = self.in_proj(x)

        q, k, v = mx.split(qkv, 3, -1)

        q = q.reshape(bsz, seqlen, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seqlen, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seqlen, self.n_head, -1).transpose(0, 2, 1, 3)

        kv_cache = self.kc_class.update_cache(input_pos, k, v, kv_cache, cache_idx)

        k, v = kv_cache

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)

        attn = attn.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)

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
