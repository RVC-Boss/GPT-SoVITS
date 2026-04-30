"""
CUDA Graph accelerated T2S decoder.
Uses PyTorch native scaled_dot_product_attention (no flash_attn dependency).
Adapted from gsvpp/AR/models/t2s_model_abc.py and t2s_model_flash_attn.py.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Dict, List, MutableSequence, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.graphs import CUDAGraph
from tqdm import tqdm

from AR.models.embedding_cudagraph import (
    SinePositionalEmbeddingNested as SinePositionalEmbedding,
)
from AR.models.embedding_cudagraph import TokenEmbedding
from AR.models.structs_cudagraph import T2SRequest, T2SResult, T2SSession

Tensor = torch.Tensor


class Sampler(nn.Module):
    def __init__(self, batch_size: int, vocab_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size

    def sample(
        self,
        logits: Tensor,
        previous_tokens: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> Tensor:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=1, index=previous_tokens, src=score)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        logits = logits / max(temperature, 1e-5)

        v, _ = torch.topk(logits, top_k)
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        q = torch.empty_like(probs).exponential_(1.0)
        idx_next = torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int32)

        return idx_next


# ─── KV Cache ────────────────────���───────────────────────────────────────────


class KVCacheNHD(nn.Module):
    def __init__(self, batch_size, max_seq_length, n_heads, head_dim):
        super().__init__()
        assert batch_size > 0
        cache_shape = (batch_size, max_seq_length, n_heads, head_dim)
        self.n_head = n_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.register_buffer(
            "k_cache", torch.zeros(size=cache_shape), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(size=cache_shape), persistent=False
        )

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        index = (
            (input_pos - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.n_head, self.head_dim)
            .to(torch.int64)
        )
        k_out = self.k_cache
        v_out = self.v_cache
        k_out.scatter_(1, index, k_val)
        v_out.scatter_(1, index, v_val)
        return k_out, v_out

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def prefill_kv(self, k_val: Tensor, v_val: Tensor, bs: int):
        self.k_cache[[bs], : k_val.shape[1]] = k_val
        self.v_cache[[bs], : v_val.shape[1]] = v_val


# ─── Attention (PyTorch native SDPA, no flash_attn) ─────────────────────────


class Attention(nn.Module):
    def __init__(self, n_head: int, hidden_dim: int):
        super().__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        assert hidden_dim % n_head == 0
        self.head_dim = hidden_dim // n_head
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(0.1)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict, prefix, *args):
        keys_to_modify = [key for key in state_dict if "in_proj_" in key]
        for key in keys_to_modify:
            new_key = key.replace("in_proj_", "in_proj.")
            state_dict[new_key] = state_dict.pop(key)

    def forward(
        self, x: Tensor, input_pos: Tensor, kv_cache: KVCacheNHD
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj.forward(x).chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)

        k_cache, v_cache = kv_cache.update(input_pos, k, v)

        q = q.transpose(1, 2)  # [B, H, 1, D]
        k_out = k_cache.transpose(1, 2)  # [B, H, max_seq, D]
        v_out = v_cache.transpose(1, 2)  # [B, H, max_seq, D]

        attn = F.scaled_dot_product_attention(q, k_out, v_out)

        attn = self.dropout.forward(attn)
        attn = attn.transpose(1, 2).reshape(bsz, seqlen, self.hidden_dim)
        attn = self.out_proj.forward(attn)
        return attn

    def prefill(self, x: Tensor, mask: Tensor, kv_cache: KVCacheNHD) -> Tensor:
        bsz = x.size(0)
        outputs = []
        for bs in range(bsz):
            x_b = x[bs].unsqueeze(0)
            q, k, v = self.in_proj.forward(x_b.unsqueeze(0)).chunk(3, dim=-1)
            q = q.contiguous().view(1, -1, self.n_head, self.head_dim)
            k = k.contiguous().view(1, -1, self.n_head, self.head_dim)
            v = v.contiguous().view(1, -1, self.n_head, self.head_dim)
            kv_cache.prefill_kv(k, v, bs)
            q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
            attn_mask = (
                mask[bs].unsqueeze(0).unsqueeze(0).expand(1, self.n_head, -1, -1)
            )
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn = self.dropout.forward(attn)
            attn = attn.transpose(1, 2).contiguous().view(1, -1, self.hidden_dim)
            output = self.out_proj.forward(attn)
            outputs.append(output.squeeze(0))
        return torch.nested.nested_tensor(outputs)


# ─── Feed Forward ────────────────────────────────────────────────────────────


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout.forward(
            self.linear2(self.dropout.forward(F.relu(self.linear1(x))))
        )


# ─── Transformer Block ──────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, n_head, ffn_dim, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = Attention(n_head, hidden_dim)
        self.feed_forward = FeedForward(hidden_dim, ffn_dim)
        self.attention_norm = nn.LayerNorm([hidden_dim])
        self.ffn_norm = nn.LayerNorm([hidden_dim])
        self.dropout = nn.Dropout(0.1)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix, *args):
        for key in list(state_dict.keys()):
            new_key = (
                key.replace("self_attn", "attention")
                .replace("linear", "feed_forward.linear")
                .replace("norm1", "attention_norm")
                .replace("norm2", "ffn_norm")
            )
            state_dict[new_key] = state_dict.pop(key)

    def forward(
        self, x: Tensor, input_pos: Tensor, kv_cache: KVCacheNHD
    ) -> Tensor:
        h = self.attention_norm.forward(
            x + self.dropout.forward(self.attention.forward(x, input_pos, kv_cache))
        )
        out = self.ffn_norm.forward(h + self.feed_forward.forward(h))
        return out

    def prefill(self, x: Tensor, mask: Tensor, kv_cache: KVCacheNHD) -> Tensor:
        h = self.attention_norm.forward(
            x + self.dropout.forward(self.attention.prefill(x, mask, kv_cache))
        )
        out = self.ffn_norm.forward(h + self.feed_forward.forward(h))
        return out


# ─── Transformer Decoder ────────────────────────────────────────────────────


class TransformerDecoder(nn.Module):
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
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        assert hidden_dim % n_head == 0
        self.head_dim = hidden_dim // n_head
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.layers = nn.ModuleList(
            TransformerBlock(n_head, ffn_dim, hidden_dim) for _ in range(n_layer)
        )
        self.max_seq_length: int = max_seq_length
        self.max_batch_size: int = max_batch_size

    def forward(
        self,
        input_pos: Tensor,
        x: Tensor,
        kv_caches: MutableSequence[KVCacheNHD],
    ):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer.forward(x, input_pos, kv_cache)
        return x

    def prefill(
        self,
        x: Tensor,
        mask: Tensor,
        kv_caches: MutableSequence[KVCacheNHD],
    ):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer.prefill(x, mask, kv_cache)
        return x


# ─── T2S Decoder ─────────────────────────────────────────────────────────────


class T2SDecoder(nn.Module):
    def __init__(
        self,
        config,
        *args,
        norm_first=False,
        max_seq_length=2500,
        max_batch_size=10,
        **kwds,
    ) -> None:
        super().__init__()
        hidden_dim = config["model"]["hidden_dim"]
        embedding_dim = config["model"]["embedding_dim"]
        n_head = config["model"]["head"]
        n_layer = config["model"]["n_layer"]
        vocab_size = config["model"]["vocab_size"]
        phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        p_dropout = config["model"]["dropout"]
        EOS = config["model"]["EOS"]
        ffn_dim = hidden_dim * 4

        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        assert hidden_dim % n_head == 0
        self.head_dim = hidden_dim // n_head
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.phoneme_vocab_size = phoneme_vocab_size
        self.p_dropout = p_dropout
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.EOS = EOS
        assert self.EOS == self.vocab_size - 1

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim, self.phoneme_vocab_size, self.p_dropout
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_length,
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim, self.vocab_size, self.p_dropout
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_length,
        )
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.h = TransformerDecoder(
            hidden_dim,
            n_layer,
            n_head,
            ffn_dim,
            vocab_size,
            max_seq_length,
            max_batch_size,
        )

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        model_keys = [key for key in state_dict if key.startswith("model.")]
        for key in model_keys:
            new_key = key[len("model.") :]
            state_dict[new_key] = state_dict.pop(key)

    def init_cache(self, bsz: int = 0) -> nn.ModuleList:
        bsz = bsz or self.h.max_batch_size
        assert bsz <= self.h.max_batch_size
        seq_lens = self.h.max_seq_length
        device = self.bert_proj.bias.device
        dtype = self.bert_proj.bias.dtype
        return nn.ModuleList(
            [
                KVCacheNHD(bsz, seq_lens, self.n_head, self.head_dim)
                for _ in range(self.n_layer)
            ],
        ).to(device, dtype)

    def embed(
        self,
        x: List[torch.Tensor],
        y: torch.Tensor,
        bert_features: List[torch.Tensor],
    ):
        x_nested = torch.nested.nested_tensor(x)
        assert x_nested.size(0) <= self.max_batch_size
        bert_features_nested = torch.nested.nested_tensor(
            list(map(lambda t: t.transpose(0, 1), bert_features))
        )
        x_emb = self.ar_text_embedding.forward(x_nested)
        bert = self.bert_proj.forward(bert_features_nested)
        x_emb = x_emb + bert
        x_pos = self.ar_text_position.prefill(x_emb)

        y_nested = torch.nested.nested_tensor(list(y.unbind(0)))
        y_emb = self.ar_audio_embedding.forward(y_nested)
        y_pos = self.ar_audio_position.prefill(y_emb)

        xy_pos = torch.nested.nested_tensor(
            [torch.cat([x_pos[i], y_pos[i]]) for i in range(len(x))]
        )
        return xy_pos

    def capture(
        self,
        input_pos: Tensor,
        x: Tensor,
        x_dec: Tensor,
        kv_caches,
    ) -> CUDAGraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        graph = torch.cuda.CUDAGraph()

        with torch.cuda.stream(s):
            for _ in range(5):
                self.h.forward(input_pos, x, kv_caches)
        torch.cuda.current_stream().wait_stream(s)

        with torch.cuda.graph(graph):
            x_dec.copy_(self.h.forward(input_pos, x, kv_caches))
        torch.cuda.synchronize()

        return graph


# ─── CUDA Graph Runner ───────────────────────────────────────────────────────


class CUDAGraphRunner:
    def __init__(
        self,
        decoder_model: T2SDecoder,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        assert device.type in {"cpu", "cuda", "mps", "xpu", "mtia"}
        assert dtype in {torch.float16, torch.bfloat16, torch.float32}
        self.device = device
        self.dtype = dtype
        self.decoder_model: T2SDecoder = decoder_model.to(self.device, self.dtype)
        self.graph: Optional[CUDAGraph] = None
        self.xy_pos_ = torch.rand(
            (1, 1, decoder_model.embedding_dim), device=device
        ).to(dtype)
        self.xy_dec_ = torch.rand(
            (1, 1, decoder_model.embedding_dim), device=device
        ).to(dtype)
        self.kv_cache = decoder_model.init_cache(1)
        self.input_pos = torch.tensor([10]).int().cuda()

    def _handle_request(self, request: T2SRequest):
        with self.device:
            for i in self.kv_cache:
                i.empty()

            decoder = self.decoder_model
            session = T2SSession(decoder, request, device=self.device, dtype=self.dtype)
            self.input_pos.copy_(session.input_pos)

            t1 = 0.0
            infer_speed = 0.0
            y = session.y
            bsz = y.size(0)

            for idx in tqdm(range(1500)):
                if idx == 0:
                    xy_dec = decoder.h.prefill(
                        session.xy_pos, session.attn_mask_nested, self.kv_cache
                    )
                    xy_dec = torch.stack([t[[-1]] for t in xy_dec.unbind()])
                else:
                    if (
                        request.use_cuda_graph
                        and self.graph is None
                        and torch.cuda.is_available()
                    ):
                        self.xy_pos_.copy_(session.xy_pos)
                        self.graph = decoder.capture(
                            self.input_pos,
                            self.xy_pos_,
                            self.xy_dec_,
                            kv_caches=self.kv_cache,
                        )

                    if self.graph:
                        self.xy_pos_.copy_(session.xy_pos)
                        self.graph.replay()
                        xy_dec = self.xy_dec_.clone()
                    else:
                        xy_dec = decoder.h.forward(
                            self.input_pos,
                            session.xy_pos,
                            self.kv_cache,
                        )

                logits = decoder.ar_predict_layer(xy_dec[:, -1])
                self.input_pos.add_(1)

                if idx == 0:
                    logits[:, -1] = float("-inf")

                samples = session.sampler.sample(
                    logits=logits,
                    previous_tokens=session.y,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    temperature=request.temperature,
                )

                session.y = torch.cat([session.y, samples], dim=1)

                argmax_token = torch.argmax(logits, dim=-1)
                sample_token = samples.squeeze(1)
                EOS_mask = (argmax_token == decoder.EOS) | (
                    sample_token == decoder.EOS
                )

                newly_done_mask = EOS_mask & (~session.completed)
                newly_done_indices = newly_done_mask.nonzero()

                if newly_done_indices.numel() > 0:
                    session.y_results[newly_done_indices[0]] = session.y[
                        newly_done_indices[0], session.y_len : -1
                    ].squeeze(0)
                    session.completed[newly_done_indices] = True

                if torch.all(session.completed).item():
                    if session.y.size(1) == 0:
                        session.y = torch.cat(
                            [session.y, torch.zeros_like(samples)], dim=1
                        )
                        tqdm.write("Bad Zero Prediction")
                    else:
                        tqdm.write(
                            f"T2S Decoding EOS {session.prefill_len.tolist().__str__().strip('[]')} -> \n"
                            f"{[i.size(0) for i in session.y_results].__str__().strip('[]')}"
                        )
                        tqdm.write(
                            f"Infer Speed: {(idx - 1) / (time.perf_counter() - t1):.2f} token/s"
                        )
                        infer_speed = (idx - 1) / (time.perf_counter() - t1)
                    break

                if (
                    request.early_stop_num != -1
                    and (session.y.size(1) - session.y_len) > request.early_stop_num
                ) or idx == 1499:
                    for i in range(bsz):
                        if not session.completed[i].item():
                            session.y_results[i] = session.y[i, session.y_len :]
                            session.completed[i] = True
                    break

                y_emb = decoder.ar_audio_embedding(session.y[:, -1:])
                session.xy_pos = decoder.ar_audio_position.forward(
                    self.input_pos - session.x_lens, y_emb
                )

                if idx == 2:
                    t1 = time.perf_counter()

                if idx % 100 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return session.y_results[: request.valid_length], infer_speed

    def generate(self, request: T2SRequest) -> T2SResult:
        try:
            result, infer_speed = self._handle_request(request)
            t2s_result = T2SResult(
                result=result, infer_speed=infer_speed, status="Success"
            )
        except Exception as e:
            t2s_result = T2SResult(
                status="Error", exception=e, traceback=traceback.format_exc()
            )
        return t2s_result

    @staticmethod
    def load_decoder(weights_path, max_batch_size=1) -> T2SDecoder:
        print(
            f"Loading Text2Semantic Weights from {weights_path} with CUDA Graph (SDPA) Implement"
        )
        dict_s1 = torch.load(
            weights_path, map_location="cpu", weights_only=False#, mmap=True
        )
        config = dict_s1["config"]
        decoder = T2SDecoder(config, max_batch_size=max_batch_size)
        state_dict = dict_s1["weight"]
        decoder.load_state_dict(state_dict)
        return decoder.eval()
