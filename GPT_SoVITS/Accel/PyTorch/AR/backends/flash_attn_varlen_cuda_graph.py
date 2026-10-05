"""
Modified From https://github.com/XXXXRT666/GPT-SoVITS
"""

import flash_attn  # type: ignore
import torch

from ... import nn
from ..structs import KVCache
from ..t2s_model_abc import (
    AttentionABC,
    FeedForward,
    KVCacheNHD,
    T2SDecoderABC,
    TransformerBlockABC,
    TransformerDecoderABC,
)


Tensor = torch.Tensor

class Attention(AttentionABC):
    def __init__(self, n_head, hidden_dim, max_seq_length):
        super().__init__(n_head, hidden_dim, max_seq_length)

        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.kv_class = KVCacheNHD

    def __call__(self, x: Tensor, input_pos: Tensor, kv_cache: KVCache, *args, **kwds) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)

        k_cache, v_cache = kv_cache

        attn: Tensor = flash_attn.flash_attn_with_kvcache(  # type: ignore
            q, k_cache, v_cache, k, v, cache_seqlens=input_pos - 1
        )

        attn = attn.view(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj(attn)

        return attn


class TransformerBlock(TransformerBlockABC):
    def __init__(self, n_head, ffn_dim, hidden_dim, max_seq_length) -> None:
        super().__init__(n_head, ffn_dim, hidden_dim, max_seq_length)

        self.attention = Attention(n_head, hidden_dim, max_seq_length)
        self.feed_forward = FeedForward(hidden_dim, ffn_dim)
        self.attention_norm = nn.LayerNorm([self.hidden_dim])
        self.ffn_norm = nn.LayerNorm([self.hidden_dim])


class TransformerDecoder(TransformerDecoderABC):
    def __init__(
        self,
        hidden_dim,
        n_layer,
        n_head,
        ffn_dim,
        vocab_size,
        max_seq_length,
        max_batch_size,
    ) -> None:
        super().__init__(hidden_dim, n_layer, n_head, ffn_dim, vocab_size, max_seq_length, max_batch_size)

        self.layers = nn.ModuleList(  # type: ignore
            TransformerBlock(n_head, ffn_dim, hidden_dim, max_seq_length) for _ in range(n_layer)
        )


class T2SDecoder(T2SDecoderABC):
    def __init__(
        self,
        config,
        max_seq_length=1024,
        max_batch_size=10,
    ) -> None:
        assert torch.cuda.is_available()
        super().__init__(config, max_seq_length, max_batch_size)

        self.h: TransformerDecoderABC = TransformerDecoder(
            self.hidden_dim,
            self.n_layer,
            self.n_head,
            self.ffn_dim,
            self.vocab_size,
            max_seq_length,
            max_batch_size,
        )

        self.kv_class = KVCacheNHD

        self.graph_applicable = True

    # Flash Attn backend keeps attn handling internal, no extra buffer work needed
    def pre_forward_slots_hook(self, slots, runner):
        return super().pre_forward_slots_hook(slots, runner)

    def post_forward_slots_hook(self, slots, runner) -> None:
        return super().post_forward_slots_hook(slots, runner)

    def bind_session_hook(self, session, runner) -> None:
        return super().bind_session_hook(session, runner)

    def unbind_session_hook(self, session, runner) -> None:
        return super().unbind_session_hook(session, runner)
