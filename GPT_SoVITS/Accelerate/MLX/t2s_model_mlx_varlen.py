from __future__ import annotations

import math
from typing import MutableSequence, cast

import mlx.core as mx
import mlx.nn as nn

from .structs_mlx import KVCacheProtocol, T2SDecoderProtocol, T2SSessionMLX

Array = mx.array


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

    @property
    def weight(self):
        return self.word_embeddings.weight

    def embedding(self, index: int):
        return self.word_embeddings.weight[index : index + 1]

    def __call__(self, x: Array):
        x = self.word_embeddings(x)
        x = self.dropout(x)
        return x


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
        scale: bool = False,
        max_batch_size: int = 10,
        max_seq_len: int = 1800,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = mx.ones(1)
        self.dropout = nn.Dropout(p=dropout)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.reverse = False
        self._pe = mx.zeros((max_batch_size, max_seq_len, embedding_dim))
        self.compute_pe()

    def compute_pe(self):
        """Reset the positional encodings."""

        if self.reverse:
            position = mx.expand_dims(mx.arange(self.max_seq_len - 1, -1, -1.0), axis=1)
        else:
            position = mx.expand_dims(mx.arange(self.max_seq_len), axis=1)
        div_term = mx.exp(
            mx.arange(
                0,
                self.embedding_dim,
                2,
            )
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pe = self._pe
        pe[:, :, 0::2] = mx.sin(position * div_term)
        pe[:, :, 1::2] = mx.cos(position * div_term)

    def __call__(self, input_pos: Array, x: Array):
        """
        Args:
            input_pos (Array): [batch_size, ]
            x (Array): [batch_size, 1, embed_dim]

        Returns:
            embedded_x (Array): [batch_size, 1, embed_dim]
        """

        batch_size = cast(tuple[int, ...], x.shape)[0]
        pe_values = self._pe[mx.arange(batch_size), input_pos - 1]  # (batch_size, embed_dim)

        return x * self.x_scale + self.alpha * mx.expand_dims(pe_values, 1)  # (batch_size, 1, embed_dim)

    def prefill(self, x: Array):
        """
        Args:
            x (Array): [batch_size, seq_len, embed_dim]

        Returns:
            embedded_x (Array): [batch_size, seq_len, embed_dim]
        """
        pe_values = self._pe[:, : cast(tuple[int, ...], x.shape)[-2]]
        return x * self.x_scale + self.alpha * pe_values


class KVCache(nn.Module, KVCacheProtocol):
    def __init__(self, batch_size: int, max_seq_length: int, n_heads: int, head_dim: int) -> None:
        super().__init__()

        self.n_head = n_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        assert batch_size > 0
        cache_shape = (batch_size, n_heads, max_seq_length, head_dim)

        self.cache_idx = mx.arange(batch_size)

        self.k_cache: Array = mx.zeros(cache_shape)
        self.v_cache: Array = mx.zeros(cache_shape)

    def empty(self):
        self.k_cache[:] = 0
        self.v_cache[:] = 0

    def update_cache(self, input_pos: Array, k_val: Array, v_val: Array):
        # input_pos: [B, ], k_val: [B, H, 1, D]

        k_out = self.k_cache
        v_out = self.v_cache

        k_out[self.cache_idx, :, input_pos, None] = k_val
        v_out[self.cache_idx, :, input_pos, None] = v_val

        return k_out, v_out

    def prefill_kv(self, k_val: Array, v_val: Array):
        # k_val: [B, S, H, D]
        self.k_cache[..., : cast(tuple[int, ...], k_val.shape)[1], :] = k_val.swapaxes(1, 2)
        self.v_cache[..., : cast(tuple[int, ...], v_val.shape)[1], :] = v_val.swapaxes(1, 2)

    def sync_cache(self, kv_cache: KVCacheProtocol):
        self.k_cache[:] = kv_cache.k_cache
        self.v_cache[:] = kv_cache.v_cache


class Attention(nn.Module):
    def __init__(self, n_head: int, hidden_dim: int, max_seq_length: int):
        super().__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        assert hidden_dim % n_head == 0
        self.head_dim = hidden_dim // n_head

        self.max_seq_length = max_seq_length

        # key, query, value projections for all heads, but in a batch
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.scale = 1 / math.sqrt(self.head_dim)

        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: Array, input_pos: Array, kv_cache: KVCacheProtocol, attn_mask: Array):
        bsz, seqlen, _ = cast(tuple[int, ...], x.shape)

        q, k, v = self.in_proj(x).split(3, axis=-1)

        q, k, v = map(lambda x: x.reshape(bsz, seqlen, self.n_head, self.head_dim), (q, k, v))

        q, k, v = map(lambda x: x.swapaxes(1, 2), (q, k, v))

        k, v = kv_cache.update_cache(input_pos, k, v)

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)

        attn = self.dropout(attn)

        attn = attn.swapaxes(1, 2).reshape(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj(attn)

        return attn

    def prefill(self, x: Array, kv_cache: KVCacheProtocol, attn_mask: Array):
        bsz, seqlen, _ = cast(tuple[int, ...], x.shape)

        q, k, v = self.in_proj(mx.expand_dims(x, 0)).split(3, axis=-1)

        q, k, v = map(lambda x: x.reshape(bsz, seqlen, self.n_head, self.head_dim), (q, k, v))

        kv_cache.prefill_kv(k, v)

        q, k, v = map(lambda x: x.swapaxes(1, 2), (q, k, v))

        attn = mx.fast.scaled_dot_product_attention(q, k, v, mask=attn_mask, scale=self.scale)

        attn = mx.nan_to_num(attn)

        attn = self.dropout(attn)

        attn = attn.swapaxes(1, 2).reshape(1, -1, self.hidden_dim)

        output = self.out_proj(attn)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: Array):
        return self.dropout(self.linear2(self.dropout(nn.relu(self.linear1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, n_head: int, ffn_dim: int, hidden_dim: int, max_seq_length: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        self.attention = Attention(n_head, hidden_dim, max_seq_length)
        self.feed_forward = FeedForward(hidden_dim, ffn_dim)
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: Array, input_pos: Array, kv_cache: KVCacheProtocol, attn_mask: Array):
        h = self.attention_norm(
            x
            + self.dropout(
                self.attention(
                    x,
                    input_pos,
                    kv_cache,
                    attn_mask,
                )
            )
        )
        out = self.ffn_norm(h + self.feed_forward(h))
        return out

    def prefill(self, x: Array, mask: Array, kv_cache: KVCacheProtocol):
        h = self.attention_norm(
            x
            + self.dropout(
                self.attention.prefill(
                    x,
                    kv_cache,
                    mask,
                )
            )
        )
        out = self.ffn_norm(h + self.feed_forward(h))

        return out


class TransformerDecoder(nn.Module):
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
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_head = n_head
        assert hidden_dim % n_head == 0

        self.head_dim = hidden_dim // n_head
        self.vocab_size = vocab_size

        self.n_layer = n_layer

        self.layers: MutableSequence[TransformerBlock] = [
            TransformerBlock(
                n_head,
                ffn_dim,
                hidden_dim,
                max_seq_length,
            )
            for _ in range(n_layer)
        ]

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

    def __call__(self, input_pos: Array, x: Array, kv_caches: MutableSequence[KVCacheProtocol], *args, **kwds):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(x, input_pos, kv_cache, *args, **kwds)
        return x

    def prefill(self, x: Array, mask: Array, kv_caches: MutableSequence[KVCacheProtocol]):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer.prefill(x, mask, kv_cache)
        return x


class T2SDecoder(nn.Module, T2SDecoderProtocol):
    def __init__(
        self,
        config: dict,
        max_seq_length: int = 1800,
        max_batch_size: int = 10,
    ) -> None:
        super().__init__()

        hidden_dim: int = config["model"]["hidden_dim"]
        embedding_dim: int = config["model"]["embedding_dim"]
        n_head: int = config["model"]["head"]
        n_layer: int = config["model"]["n_layer"]
        vocab_size: int = config["model"]["vocab_size"]
        phoneme_vocab_size: int = config["model"]["phoneme_vocab_size"]
        p_dropout: float = config["model"]["dropout"]
        EOS: int = config["model"]["EOS"]
        ffn_dim: int = hidden_dim * 4

        self.n_layer = int(n_layer)
        self.hidden_dim = int(hidden_dim)
        self.n_head = int(n_head)
        assert hidden_dim % n_head == 0

        self.head_dim = int(hidden_dim // n_head)
        self.embedding_dim = int(embedding_dim)
        self.ffn_dim = int(ffn_dim)
        self.vocab_size = int(vocab_size)
        self.phoneme_vocab_size = int(phoneme_vocab_size)
        self.p_dropout = float(p_dropout)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.EOS = EOS
        assert self.EOS == self.vocab_size - 1

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.h = TransformerDecoder(
            self.hidden_dim, self.n_layer, self.n_head, self.ffn_dim, self.vocab_size, max_seq_length, max_batch_size
        )

        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_length,
        )
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_length,
        )

    def init_cache(self, bsz: int = 0) -> MutableSequence[KVCacheProtocol]:
        bsz = bsz or self.h.max_batch_size
        assert bsz <= self.h.max_batch_size
        seq_lens = self.h.max_seq_length
        dtype = self.bert_proj.bias.dtype
        cache: MutableSequence[KVCacheProtocol] = [
            KVCache(bsz, seq_lens, self.n_head, self.head_dim) for _ in range(self.n_layer)
        ]
        for c in cache:
            cast(KVCache, c).set_dtype(dtype)
        mx.eval(cache)
        return cache

    def embed(
        self,
        x: list[Array],
        y: Array,
        bert_features: list[Array],
    ):
        x_len: list[int] = [cast(tuple[int, ...], i.shape)[0] for i in x]
        x_len_max = max(x_len)
        xy_pos = mx.zeros((len(x), x_len_max + cast(tuple[int, ...], y.shape)[1], self.embedding_dim)).astype(
            bert_features[0].dtype
        )

        bert_features = list(map(lambda x: x.swapaxes(0, 1), bert_features))

        y_len = cast(tuple[int, ...], y.shape)[1]
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position.prefill(y_emb)

        for bs, (x_, len_, bert_feature) in enumerate(zip(x, x_len, bert_features)):
            x_emb = self.ar_text_embedding(x_)
            bert = self.bert_proj(bert_feature)
            x_emb = x_emb + bert
            x_pos = self.ar_text_position.prefill(mx.expand_dims(x_emb, 0))
            xy_pos[[bs], :len_] = x_pos
            xy_pos[[bs], len_ : len_ + y_len] = y_pos

        mx.eval(xy_pos)
        return xy_pos

    # def compile(self):
    # setattr(self.h, "__call__", mx.compile(self.h.__call__))
    # setattr(self.h, "prefill", mx.compile(self.h.prefill, shapeless=True))

    def pre_forward(self, session: T2SSessionMLX):
        attn_mask = session.attn_mask
        return list(), dict(attn_mask=attn_mask)

    def post_forward(self, idx: int, session: T2SSessionMLX) -> None:
        if idx == 0:
            prefill_len = session.prefill_len
            bsz = session.bsz

            range_tensor = mx.arange(self.max_seq_length).reshape(1, 1, 1, self.max_seq_length)
            prefill_len_expanded = prefill_len.reshape(bsz, 1, 1, 1)
            attn_mask = range_tensor < prefill_len_expanded
            attn_mask = mx.repeat(attn_mask, self.n_head, 1)

            session.attn_mask = attn_mask

        attn_mask = session.attn_mask
        input_pos = session.input_pos
        attn_mask[mx.arange(session.bsz), :, :, input_pos] = True
        mx.eval(attn_mask)
