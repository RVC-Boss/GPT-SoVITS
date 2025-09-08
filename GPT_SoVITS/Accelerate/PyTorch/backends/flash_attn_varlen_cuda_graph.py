"""
Modified From https://github.com/XXXXRT666/GPT-SoVITS
"""

from typing import Dict, List, Tuple

import kernels
import torch

from .. import nn
from ..structs import T2SSession
from ..t2s_model_abc import (
    AttentionABC,
    CUDAGraphCacheABC,
    FeedForward,
    KVCacheNHD,
    KVCacheProtocol,
    T2SDecoderABC,
    TransformerBlockABC,
    TransformerDecoderABC,
)

flash_attn_kernel = None
try:
    import flash_attn_interface as flash_attn  # type: ignore

    flash_attn_kernel = flash_attn.flash_attn_with_kvcache
except ModuleNotFoundError:
    try:
        import flash_attn  # type: ignore

        flash_attn_kernel = flash_attn.flash_attn_with_kvcache

    except ModuleNotFoundError:
        pass

if flash_attn_kernel is None:
    flash_attn_kernel = kernels.get_kernel("kernels-community/flash-attn").flash_attn_with_kvcache


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
        max_seq_length=2000,
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

    def post_forward(self, idx: int, session: T2SSession) -> None:
        return super().post_forward(idx, session)

    def pre_forward(self, session: T2SSession) -> Tuple[List, Dict]:
        return super().pre_forward(session)


class CUDAGraphCache(CUDAGraphCacheABC):
    def __init__(
        self,
        decoder: T2SDecoder,
    ) -> None:
        self.is_applicable = True
        super().__init__(decoder)

    def release_graph(self, session: T2SSession):
        if session.id == self.id:
            self.assigned = False
        else:
            del session.graph, session.xy_pos_, session.xy_dec_, session.input_pos, session.kv_cache

    def get_cache_graph(self, session: T2SSession):
        assert self.graph
        session.graph = self.graph
        session.stream = self.stream

        session.xy_pos_ = self.xy_pos
        session.xy_dec_ = self.xy_dec
        session.input_pos = self.input_pos.copy_(session.input_pos)

        for cache, cache_ in zip(self.kv_cache, session.kv_cache):
            cache.sync_cache(cache_)

    def capture_new_graph(self, session: T2SSession):
        session.xy_pos_ = self.xy_pos.clone()
        session.xy_dec_ = self.xy_dec.clone()
        session.input_pos = self.input_pos.clone().copy_(session.input_pos)

        args, kwds = self.decoder.pre_forward(session)
        graph = self.decoder.capture(self.input_pos, self.xy_pos, self.xy_dec, self.kv_cache, *args, **kwds)
        session.graph = graph
        session.stream = torch.cuda.Stream()  # type: ignore
