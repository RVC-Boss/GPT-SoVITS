from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal, MutableSequence, Type

import mlx.core as mx
import mlx.nn as nn
from mlx.core import Dtype

from .structs_mlx import KVCache, KVCacheProtocol, T2SDecoderProtocol, T2SSessionMLX

Array = mx.array


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

    @property
    def weight(self):
        return self.word_embeddings.weight

    def embedding(self, index: int):
        return self.word_embeddings.weight[index : index + 1]

    def __call__(self, x: Array):
        x = self.word_embeddings(x)
        return x


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        scale: bool = False,
        max_batch_size: int = 10,
        max_seq_length: int = 1500,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = mx.ones(1)
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length

        self.reverse = False
        self.pe: Array | None = None

    def compute_pe(self, dtype: Dtype):
        """Compute the positional encodings."""

        if self.pe is not None and self.pe.dtype == dtype:
            return

        if self.reverse:
            position = mx.expand_dims(mx.arange(self.max_seq_length - 1, -1, -1.0), axis=1)
        else:
            position = mx.expand_dims(mx.arange(self.max_seq_length), axis=1)
        div_term = mx.exp(
            mx.arange(
                0,
                self.embedding_dim,
                2,
            )
            * -(math.log(10000.0) / self.embedding_dim)
        ).astype(dtype)
        pe = mx.zeros((self.max_batch_size, self.max_seq_length, self.embedding_dim)).astype(dtype)

        pe[:, :, 0::2] = mx.sin(position * div_term).astype(dtype)
        pe[:, :, 1::2] = mx.cos(position * div_term).astype(dtype)

        self.pe = pe

    def __call__(self, input_pos: Array, x: Array):
        """
        Args:
            input_pos (Array): [batch_size, ]
            x (Array): [batch_size, 1, embed_dim]

        Returns:
            embedded_x (Array): [batch_size, 1, embed_dim]
        """
        self.compute_pe(x.dtype)
        assert self.pe is not None

        batch_size = x.shape[0]
        pe_values = self.pe[mx.arange(batch_size), input_pos - 1]  # (batch_size, embed_dim)

        return x * self.x_scale + self.alpha * mx.expand_dims(pe_values, 1)  # (batch_size, 1, embed_dim)

    def prefill(self, x: Array):
        """
        Args:
            x (Array): [batch_size, seq_len, embed_dim]

        Returns:
            embedded_x (Array): [batch_size, seq_len, embed_dim]
        """
        self.compute_pe(x.dtype)
        assert self.pe is not None

        batch_size = x.shape[0]
        pe_values = self.pe[mx.arange(batch_size), : x.shape[-2]]
        return x * self.x_scale + self.alpha * pe_values


class KVCacheHND(KVCacheProtocol):
    @staticmethod
    def empty(kv_cache):
        assert len(kv_cache) == 2
        k_cache, v_cache = kv_cache

        k_cache[:] = 0
        v_cache[:] = 0

    @staticmethod
    def update_cache(input_pos, k_val, v_val, kv_cache, cache_idx):
        # input_pos: [B, ], k_val: [B, H, 1, D]
        assert len(kv_cache) == 2
        k_out, v_out = kv_cache
        ip0 = input_pos - 1

        k_out[cache_idx, :, ip0, None] = k_val
        v_out[cache_idx, :, ip0, None] = v_val

        return k_out, v_out

    @staticmethod
    def prefill_kv(k_val, v_val, kv_cache):
        # k_val: [B, H, S, D]
        assert len(kv_cache) == 2
        k_cache, v_cache = kv_cache

        k_cache[..., : k_val.shape[2], :] = k_val
        v_cache[..., : v_val.shape[2], :] = v_val

    @staticmethod
    def init_cache(batch_size: int, max_seq_length: int, n_heads: int, head_dim: int, dtype: mx.Dtype) -> KVCache:
        cache_shape = (batch_size, n_heads, max_seq_length, head_dim)

        return (mx.zeros(cache_shape, dtype=dtype), mx.zeros(cache_shape, dtype=dtype))


class AttentionABC(ABC, nn.Module):
    def __init__(self, n_head: int, hidden_dim: int, max_seq_length: int, *args, **kwds):
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

        self.kc_class: KVCacheProtocol

    @abstractmethod
    def __call__(self, x: Array, input_pos: Array, kv_cache: KVCache, cache_idx: Array, attn_mask: Array) -> Array: ...

    def prefill(self, x: Array, kv_cache: KVCache, attn_mask: Array):
        bsz, seqlen, _ = x.shape

        qkv = self.in_proj(x)

        q, k, v = mx.split(qkv, 3, -1)

        q = q.reshape(bsz, seqlen, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seqlen, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seqlen, self.n_head, -1).transpose(0, 2, 1, 3)

        self.kc_class.prefill_kv(k, v, kv_cache)

        attn = mx.fast.scaled_dot_product_attention(q, k, v, mask=attn_mask, scale=self.scale)

        attn = mx.nan_to_num(attn)

        attn = attn.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)

        output = self.out_proj(attn)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: Array):
        return self.linear2(nn.relu(self.linear1(x)))


class TransformerBlockABC(nn.Module):
    def __init__(self, n_head: int, ffn_dim: int, hidden_dim: int, max_seq_length: int, *args, **kwds) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        self.attention: AttentionABC

        self.feed_forward = FeedForward(hidden_dim, ffn_dim)
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)

    def __call__(self, x: Array, input_pos: Array, kv_cache: KVCache, cache_idx: Array, attn_mask: Array):
        h = self.attention_norm(
            x
            + self.attention(
                x,
                input_pos,
                kv_cache,
                cache_idx,
                attn_mask,
            )
        )
        out = self.ffn_norm(h + self.feed_forward(h))
        return out

    def prefill(self, x: Array, attn_mask: Array, kv_cache: KVCache):
        h = self.attention_norm(
            x
            + self.attention.prefill(
                x,
                kv_cache,
                attn_mask,
            )
        )
        out = self.ffn_norm(h + self.feed_forward(h))

        return out


class TransformerDecoderABC(nn.Module):
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
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_head = n_head
        assert hidden_dim % n_head == 0

        self.head_dim = hidden_dim // n_head
        self.vocab_size = vocab_size

        self.n_layer = n_layer

        self.layers: MutableSequence[TransformerBlockABC]

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

    def __call__(
        self,
        input_pos: Array,
        x: Array,
        kv_caches: MutableSequence[KVCache],
        cache_idx: Array,
        *args,
        **kwds,
    ):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(
                x,
                input_pos,
                kv_cache,
                cache_idx,
                *args,
                **kwds,
            )

        return x

    def prefill(self, x: Array, mask: Array, kv_caches: MutableSequence[KVCache]):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer.prefill(
                x,
                mask,
                kv_cache,
            )
        return x


class T2SDecoderABC(nn.Module, T2SDecoderProtocol):
    def __init__(
        self,
        config: dict,
        max_seq_length: int = 1500,
        max_batch_size: int = 10,
    ) -> None:
        super().__init__()

        hidden_dim: int = config["model"]["hidden_dim"]
        embedding_dim: int = config["model"]["embedding_dim"]
        n_head: int = config["model"]["head"]
        n_layer: int = config["model"]["n_layer"]
        vocab_size: int = config["model"]["vocab_size"]
        phoneme_vocab_size: int = config["model"]["phoneme_vocab_size"]
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
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.EOS = EOS
        assert self.EOS == self.vocab_size - 1

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.h: TransformerDecoderABC

        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size)
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            scale=False,
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
        )
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size)
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            scale=False,
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
        )

        self.kv_class: Type[KVCacheProtocol]

        self.bits: int = -1
        self.group_size: int = -1

    def init_cache(self, bsz: int = 0, *args, **kwds) -> MutableSequence[KVCache]:
        bsz = bsz or self.h.max_batch_size
        assert bsz <= self.h.max_batch_size
        seq_lens = self.h.max_seq_length
        dtype = self.bert_proj.bias.dtype
        cache: MutableSequence[KVCache] = [
            self.kv_class.init_cache(bsz, seq_lens, self.n_head, self.head_dim, dtype, *args, **kwds)
            for _ in range(self.n_layer)
        ]
        mx.eval(cache)
        return cache

    def embed(
        self,
        x: list[Array],
        y: Array,
        bert_features: list[Array],
    ):
        x_len: list[int] = [i.shape[0] for i in x]
        x_len_max = max(x_len)
        xy_pos = mx.zeros((len(x), x_len_max + y.shape[1], self.embedding_dim)).astype(bert_features[0].dtype)

        bert_features = list(map(lambda x: x.swapaxes(0, 1), bert_features))

        y_len = y.shape[1]
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

    def compile(self):
        setattr(self.h, "__call__", mx.compile(self.h.__call__, shapeless=True))

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

            session.attn_mask = attn_mask

        attn_mask = session.attn_mask
        input_pos = session.input_pos
        attn_mask[mx.arange(session.bsz), :, :, input_pos] = True

    def quantize(self, mode: Literal["Affine", "MXFP4"] | None = None) -> None:
        if mode is None:
            return
        if mode not in {"Affine", "MXFP4"}:
            raise ValueError(f"Unsupported quantization mode: {mode}")
        match mode:
            case "Affine":
                self.bits = 8
                self.group_size = 32
                nn.quantize(self.h, group_size=self.group_size, bits=self.bits, mode="affine")
            case "MXFP4":
                self.bits = 4
                self.group_size = 32
                nn.quantize(self.h, group_size=self.group_size, bits=self.bits, mode="mxfp4")

            case _:
                raise ValueError(f"Unsupported Quantization Mode for MLX: {mode}")
