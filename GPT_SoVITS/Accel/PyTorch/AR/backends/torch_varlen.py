from typing import NoReturn

import torch
from torch.nn import functional as F

from ... import nn
from ..structs import KVCacheProtocol, T2SSession
from ..t2s_model_abc import (
    AttentionABC,
    CUDAGraphCacheABC,
    FeedForward,
    KVCacheHNDVarlen,
    T2SDecoderABC,
    TransformerBlockABC,
    TransformerDecoderABC,
)

Tensor = torch.Tensor


class Attention(AttentionABC):
    def __init__(self, n_head, hidden_dim, max_seq_length):
        super().__init__(n_head, hidden_dim, max_seq_length)

        # key, query, value projections for all heads, but in a batch
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x: Tensor, input_pos: Tensor, kv_cache: KVCacheProtocol, attn_mask: Tensor, max_idx: Tensor):
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        k, v = kv_cache.update(input_pos, k, v)

        q, k, v = map(lambda x: x[..., :max_idx, :], (q, k, v))

        mask = attn_mask[..., :max_idx]

        attn = F.scaled_dot_product_attention(q, k, v, mask)

        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj(attn)

        return attn


class TransformerBlock(TransformerBlockABC):
    def __init__(self, n_head: int, ffn_dim: int, hidden_dim: int, max_seq_length: int) -> None:
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
        super().__init__(config, max_seq_length, max_batch_size)

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.h: TransformerDecoderABC = TransformerDecoder(
            self.hidden_dim, self.n_layer, self.n_head, self.ffn_dim, self.vocab_size, max_seq_length, max_batch_size
        )

        self.kv_class = KVCacheHNDVarlen

        self.graph_cache_class = CUDAGraphCache

    def capture(
        self,
        *args,
        **kwds,
    ) -> NoReturn:
        raise NotImplementedError("Cuda Graph Is Not Supported For Varlen Model")

    def pre_forward(self, session: T2SSession):
        attn_mask = session.attn_mask
        max_idx = session.input_pos.max()
        return list(), dict(attn_mask=attn_mask, max_idx=max_idx)

    def post_forward(self, idx: int, session: T2SSession) -> None:
        if idx == 0:
            prefill_len = session.prefill_len
            bsz = session.bsz

            range_tensor = torch.arange(self.max_seq_length).view(1, 1, 1, self.max_seq_length)
            prefill_len_expanded = prefill_len.view(bsz, 1, 1, 1)
            attn_mask = range_tensor < prefill_len_expanded

            session.attn_mask = attn_mask

        attn_mask = session.attn_mask
        input_pos = session.input_pos
        attn_mask[torch.arange(session.bsz), :, :, input_pos] = True


class CUDAGraphCache(CUDAGraphCacheABC):
    is_applicable = False

    def __init__(
        self,
        decoder,
        cache_size: int,
    ) -> None:
        super().__init__(decoder, cache_size)

    def create_graph_cache(self, bsz: int) -> NoReturn:
        raise NotImplementedError("Cuda Graph Is Not Supported For Varlen Model")
