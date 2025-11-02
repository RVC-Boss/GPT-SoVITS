"""
Modified From https://github.com/XXXXRT666/GPT-SoVITS
"""

from typing import Dict, List, Tuple

import flash_attn  # type: ignore
import torch

from ... import nn
from ..structs import T2SSession
from ..t2s_model_abc import (
    AttentionABC,
    CUDAGraphCacheABC,
    CUDAGraphStateABC,
    FeedForward,
    KVCacheNHD,
    KVCacheProtocol,
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

    def __call__(self, x: Tensor, input_pos: Tensor, kv_cache: KVCacheProtocol, *args, **kwds) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)

        attn: Tensor = flash_attn.flash_attn_with_kvcache(  # type: ignore
            q, kv_cache.k_cache, kv_cache.v_cache, k, v, cache_seqlens=input_pos - 1
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
        max_seq_length=1500,
        max_batch_size=10,
    ) -> None:
        assert torch.cuda.is_available()
        super().__init__(config, max_seq_length, max_batch_size)

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.h: TransformerDecoderABC = TransformerDecoder(
            self.hidden_dim, self.n_layer, self.n_head, self.ffn_dim, self.vocab_size, max_seq_length, max_batch_size
        )

        self.kv_class = KVCacheNHD

        self.graph_cache_class = CUDAGraphCache

    def compile(self, *args, **kwds):
        pass

    def post_forward(self, idx: int, session: T2SSession) -> None:
        return super().post_forward(idx, session)

    def pre_forward(self, session: T2SSession) -> Tuple[List, Dict]:
        return super().pre_forward(session)


class CUDAGraphState(CUDAGraphStateABC):
    applicable: bool = True

    def __init__(
        self,
        bsz: int,
        decoder: T2SDecoderABC,
    ) -> None:
        super().__init__(bsz, decoder)

    def capture(self):
        graph = self.decoder.capture(
            self.input_pos,
            self.xy_pos,
            self.xy_dec,
            self.kv_cache,
        )
        self.graph = graph
        self.stream = torch.cuda.Stream()


class CUDAGraphCache(CUDAGraphCacheABC):
    is_applicable = True

    def __init__(
        self,
        decoder,
        cache_size: int = 5,
    ) -> None:
        super().__init__(decoder, cache_size)

    def create_graph_cache(self, bsz: int):
        for _ in range(self.cache_size):
            state = CUDAGraphState(bsz, self.decoder)
            state.capture()
            self.graph_cache[bsz].put(state)
