from __future__ import annotations

import mlx.core as mx

from ..structs_mlx import KVCache
from ..t2s_model_abc_mlx import (
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
        self.kv_class = KVCacheHND

    def __call__(self, x: Array, input_pos: Array, max_idx: int, kv_cache: KVCache, attn_mask: Array):
        bsz, seqlen, _ = x.shape

        qkv = self.in_proj(x)

        q, k, v = mx.split(qkv, 3, -1)

        q = q.reshape(bsz, seqlen, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seqlen, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seqlen, self.n_head, -1).transpose(0, 2, 1, 3)

        kv_cache = self.kv_class.update_cache(input_pos, k, v, kv_cache)

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
        max_seq_length: int = 1024,
        max_batch_size: int = 10,
    ) -> None:
        super().__init__(config, max_seq_length, max_batch_size)

        self.h = TransformerDecoder(
            self.hidden_dim, self.n_layer, self.n_head, self.ffn_dim, self.vocab_size, max_seq_length, max_batch_size
        )

        self.kv_class = KVCacheHND

        self.extra_buffer_factory = lambda max_bs, dec: {
            "attn_mask": mx.zeros((max_bs, dec.n_head, 1, dec.max_seq_length), dtype=mx.bool_)
        }

    def pre_forward_slots_hook(self, slots, runner):
        attn_mask_buf = runner.extra_buffers["attn_mask_buf"]
        max_idx = runner.input_pos_buf[slots].long().max().item()
        return {"attn_mask": attn_mask_buf[slots], "max_idx": max_idx}

    def post_forward_slots_hook(self, slots, runner) -> None:
        attn_mask_buf = runner.extra_buffers["attn_mask_buf"]
        pos = runner.input_pos_buf[slots].long()
        attn_mask_buf[slots, :, :, pos] = True

    def bind_session_hook(self, session, runner) -> None:
        slots = session.slot_indices

        prefill_len = session.prefill_len
        bsz = session.bsz

        range_tensor = mx.arange(self.max_seq_length).reshape(1, 1, 1, self.max_seq_length)
        prefill_len_expanded = prefill_len.reshape(bsz, 1, 1, 1)
        attn_mask = range_tensor < prefill_len_expanded

        runner.extra_buffers["attn_mask_buf"][slots] = attn_mask

    def unbind_session_hook(self, session, runner) -> None:
        slots = session.slot_indices
        attn_mask_buf = runner.extra_buffers["attn_mask_buf"]
        attn_mask_buf[slots] = 0
