import torch
from torch.nn import functional as F

from ... import nn
from ..structs import KVCache
from ..t2s_model_abc import (
    AttentionABC,
    FeedForward,
    KVCacheHND,
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

        self.kv_class = KVCacheHND

    def __call__(self, x: Tensor, input_pos: Tensor, kv_cache: KVCache, attn_mask: Tensor):
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        k, v = self.kv_class.update(input_pos, k, v, kv_cache)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

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
        max_seq_length=1024,
        max_batch_size=10,
    ) -> None:
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

        self.kv_class = KVCacheHND

        self.extra_buffer_factory = lambda max_bs, dec: {
            "attn_mask": torch.zeros((max_bs, dec.n_head, 1, dec.max_seq_length), device=dec.device, dtype=torch.bool)
        }

        self.graph_applicable = True

    def pre_forward_slots_hook(self, slots, runner):
        mask = runner.extra_buffers["attn_mask"]
        positions = torch.arange(self.max_seq_length, device=runner.device)
        mask[:len(slots)].copy_(
            (positions[None, :] < runner.input_pos_buf[:len(slots), None])[:, None, None, :]
        )
        return {"attn_mask": mask[:len(slots)]}

    def post_forward_slots_hook(self, slots, runner) -> None:
        return

    def bind_session_hook(self, session, runner) -> None:
        return

    def unbind_session_hook(self, session, runner) -> None:
        return
