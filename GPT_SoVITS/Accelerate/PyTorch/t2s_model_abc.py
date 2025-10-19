"""
Modified From https://github.com/XXXXRT666/GPT-SoVITS
"""

from __future__ import annotations

import math
import os
import pickle
import platform
import random
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, MutableSequence

import torch
import torch._inductor.config
import torch.nn.functional as F
from torch.cuda.graphs import CUDAGraph
from torch.profiler import ExecutionTraceObserver, ProfilerAction, tensorboard_trace_handler

from tools.my_utils import get_machine_id

from . import nn
from .quantization import replace_all_linear_with_fp8
from .structs import KVCacheProtocol, T2SDecoderProtocol, T2SSession

Tensor = torch.Tensor


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
    def weight(self) -> Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def __call__(self, x: Tensor):
        x = self.word_embeddings(x)
        return x


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        scale: bool = False,
        alpha: bool = False,
        max_batch_size: int = 10,
        max_seq_length: int = 1500,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length

        self.reverse = False
        self.register_buffer("pe", torch.zeros(max_batch_size, max_seq_length, embedding_dim), persistent=False)
        self.pe: torch.Tensor
        self.compute_pe()

    def compute_pe(self):
        """Reset the positional encodings."""
        if self.reverse:
            position = torch.arange(self.max_seq_length - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(self.max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / self.embedding_dim)
        )
        pe = self.pe
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

    def __call__(self, input_pos: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            input_pos (Tensor): [batch_size, ]
            x (Tensor): [batch_size, 1, embed_dim]

        Returns:
            embedded_x (Tensor): [batch_size, 1, embed_dim]
        """

        batch_size = x.shape[0]
        pe_values = self.pe[torch.arange(batch_size), input_pos - 1]  # (batch_size, embed_dim)

        return x * self.x_scale + self.alpha * pe_values.unsqueeze(1)  # (batch_size, 1, embed_dim)

    def prefill(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [batch_size, seq_len, embed_dim]

        Returns:
            embedded_x (Tensor): [batch_size, seq_len, embed_dim]
        """

        batch_size = x.shape[0]
        pe_values = self.pe[:batch_size, : x.shape[-2]]
        return x * self.x_scale + self.alpha * pe_values


class KVCacheABC(nn.Module, ABC, KVCacheProtocol):
    def __init__(self, batch_size: int, max_seq_length: int, n_heads: int, head_dim: int) -> None:
        super().__init__()

        self.n_head = n_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self.k_cache: Tensor
        self.v_cache: Tensor

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    @abstractmethod
    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor, *args, **kwds) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def prefill_kv(self, k_val: Tensor, v_val: Tensor) -> None: ...

    def sync_cache(self, kv_cache: KVCacheProtocol):
        self.k_cache.copy_(kv_cache.k_cache)
        self.v_cache.copy_(kv_cache.v_cache)


class KVCacheNHD(KVCacheABC):
    def __init__(self, batch_size, max_seq_length, n_heads, head_dim):
        super().__init__(batch_size, max_seq_length, n_heads, head_dim)

        assert batch_size > 0
        cache_shape = (batch_size, max_seq_length, n_heads, head_dim)

        self.register_buffer("k_cache", torch.zeros(size=cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(size=cache_shape), persistent=False)

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        # input_pos: [B, ], k_val: [B, 1, H, D]

        index = (
            (input_pos - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                self.n_head,
                self.head_dim,
            )
            .to(torch.int64)
        )  # (bs, 1, num_head, head_dim)

        k_out = self.k_cache
        v_out = self.v_cache
        k_out.scatter_(1, index, k_val)
        v_out.scatter_(1, index, v_val)

        return k_out, v_out

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def prefill_kv(self, k_val: Tensor, v_val: Tensor):
        # input_pos: int, k_val: [B, S, H, D]

        self.k_cache[:, : k_val.shape[1]] = k_val
        self.v_cache[:, : v_val.shape[1]] = v_val


class KVCacheHND(KVCacheABC):
    def __init__(self, batch_size, max_seq_length, n_heads, head_dim):
        super().__init__(batch_size, max_seq_length, n_heads, head_dim)

        cache_shape = (batch_size, n_heads, max_seq_length, head_dim)

        self.register_buffer("k_cache", torch.zeros(size=cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(size=cache_shape), persistent=False)

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        # input_pos: [B, ], k_val: [B, H, 1, D]

        index = (
            (input_pos - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                self.n_head,
                -1,
                self.head_dim,
            )
            .to(torch.int64)
        )  # (bs, num_head, 1, head_dim)

        k_out = self.k_cache
        v_out = self.v_cache
        k_out.scatter_(2, index, k_val)
        v_out.scatter_(2, index, v_val)

        return k_out, v_out

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def prefill_kv(self, k_val: Tensor, v_val: Tensor):
        # input_pos: int, k_val: [B, S, H, D]

        self.k_cache[..., : k_val.shape[1], :] = k_val.transpose(1, 2)
        self.v_cache[..., : v_val.shape[1], :] = v_val.transpose(1, 2)


class KVCacheHNDVarlen(KVCacheABC):
    def __init__(self, batch_size, max_seq_length, n_heads, head_dim):
        super().__init__(batch_size, max_seq_length, n_heads, head_dim)

        cache_shape = (batch_size, n_heads, max_seq_length, head_dim)
        self.cache_idx: Tensor

        self.register_buffer("cache_idx", torch.arange(batch_size), persistent=False)
        self.register_buffer("k_cache", torch.zeros(size=cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(size=cache_shape), persistent=False)

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        # input_pos: [B, ], k_val: [B, H, 1, D]

        k_out = self.k_cache
        v_out = self.v_cache

        ip0 = input_pos - 1

        k_out[self.cache_idx, :, ip0, None] = k_val
        v_out[self.cache_idx, :, ip0, None] = v_val

        return k_out, v_out

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def prefill_kv(self, k_val: Tensor, v_val: Tensor):
        # input_pos: int, k_val: [B, S, H, D]

        self.k_cache[..., : k_val.shape[1], :] = k_val.transpose(1, 2)
        self.v_cache[..., : v_val.shape[1], :] = v_val.transpose(1, 2)


class AttentionABC(nn.Module, ABC):
    def __init__(self, n_head: int, hidden_dim: int, max_seq_length: int):
        super().__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        assert hidden_dim % n_head == 0
        self.head_dim = hidden_dim // n_head

        self.max_seq_length = max_seq_length

        # key, query, value projections for all heads, but in a batch
        self.in_proj: nn.Linear
        self.out_proj: nn.Linear

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix, *args):
        keys_to_modify = [key for key in state_dict if "in_proj_" in key]
        for key in keys_to_modify:
            new_key = key.replace("in_proj_", "in_proj.")  # in_proj_ -> in_proj.
            state_dict[new_key] = state_dict.pop(key)

    @abstractmethod
    def __call__(self, x: Tensor, input_pos: Tensor, kv_cache: KVCacheProtocol, *args, **kwds) -> Tensor: ...

    def prefill(self, x: Tensor, kv_cache: KVCacheProtocol, attn_mask: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q, k, v = map(lambda x: x.contiguous().view(bsz, seqlen, self.n_head, self.head_dim), (q, k, v))

        kv_cache.prefill_kv(k, v)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).contiguous().view(1, -1, self.hidden_dim)

        output = self.out_proj(attn)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: Tensor):
        return self.linear2(F.relu(self.linear1(x), inplace=True))


class TransformerBlockABC(nn.Module, ABC):
    def __init__(self, n_head: int, ffn_dim: int, hidden_dim: int, max_seq_length: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        self.attention: AttentionABC
        self.feed_forward: FeedForward
        self.attention_norm: nn.LayerNorm
        self.ffn_norm: nn.LayerNorm

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

    def __call__(self, x: Tensor, input_pos: Tensor, kv_cache: KVCacheProtocol, *args, **kwds):
        h = self.attention_norm(
            x
            + self.attention(
                x,
                input_pos,
                kv_cache,
                *args,
                **kwds,
            )
        )
        out = self.ffn_norm(h + self.feed_forward(h))
        return out

    def prefill(
        self,
        x: Tensor,
        kv_cache: KVCacheProtocol,
        attn_mask: Tensor,
    ) -> Tensor:
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


class TransformerDecoderABC(nn.Module, ABC):
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

        self.layers: MutableSequence[TransformerBlockABC]

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

    def __call__(self, input_pos: Tensor, x: Tensor, kv_caches: MutableSequence[KVCacheProtocol], *args, **kwds):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(x, input_pos, kv_cache, *args, **kwds)
        return x

    def prefill(self, x: Tensor, kv_caches: MutableSequence[KVCacheProtocol], attn_mask: Tensor):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer.prefill(x, kv_cache, attn_mask)
        return x


class T2SDecoderABC(nn.Module, ABC, T2SDecoderProtocol):
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

        self.bert_proj: nn.Linear
        self.ar_predict_layer: nn.Linear
        self.h: TransformerDecoderABC

        self.kv_class: type[KVCacheABC]

        self.GraphCache: CUDAGraphCacheABC | None
        self.compiled: bool = False

        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size)
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            scale=False,
            alpha=True,
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
        )
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size)
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            scale=False,
            alpha=True,
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
        )

        self.bits: int
        self.group_size: int

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix, *args):
        model_keys = [key for key in state_dict if key.startswith("model.")]
        for key in model_keys:
            new_key = key[len("model.") :]
            state_dict[new_key] = state_dict.pop(key)

    def init_cache(self, bsz: int = 0) -> MutableSequence[KVCacheProtocol]:
        bsz = bsz or self.h.max_batch_size
        assert bsz <= self.h.max_batch_size
        seq_lens = self.h.max_seq_length
        dtype = self.bert_proj.bias.dtype
        kvclass = self.kv_class

        return nn.ModuleList(
            [kvclass(bsz, seq_lens, self.n_head, self.head_dim) for _ in range(self.n_layer)],
        ).to(self.device, dtype)  # type: ignore

    def embed(
        self,
        x: list[torch.Tensor],
        y: torch.Tensor,
        bert_features: list[torch.Tensor],
    ):
        x_len: list[int] = [i.shape[0] for i in x]
        x_len_max = max(x_len)
        xy_pos = torch.zeros((len(x), x_len_max + y.shape[1], self.embedding_dim)).to(bert_features[0].dtype)

        bert_features = list(map(lambda x: x.transpose(0, 1), bert_features))

        y_len = y.shape[1]
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position.prefill(y_emb)

        for bs, (x_, len_, bert_feature) in enumerate(zip(x, x_len, bert_features)):
            x_emb = self.ar_text_embedding(x_)
            bert = self.bert_proj(bert_feature)
            x_emb = x_emb + bert
            x_pos = self.ar_text_position.prefill(x_emb.unsqueeze(0))
            xy_pos[[bs], :len_] = x_pos
            xy_pos[[bs], len_ : len_ + y_len] = y_pos

        return xy_pos

    def compile(self, *args, **kwds):
        if (
            torch.cuda.is_available()
            and platform.system() != "Windows"
            or platform.system() == "macOS"
            and self.compiled is False
        ):
            cache_path = Path.cwd() / "compile_cache"
            if cache_path.exists() is False:
                cache_path.mkdir(parents=True, exist_ok=True)
            else:
                assert cache_path.is_dir()
                cache_file = (
                    cache_path
                    / f"t2s_decoder_{self.n_layer}_{self.hidden_dim}_{self.n_head}_{self.ffn_dim}_{self.phoneme_vocab_size}_{get_machine_id()}_{torch.__version__}.GSV"
                )
                if cache_file.exists():
                    try:
                        with open(cache_file, "rb") as f:
                            cache_data = pickle.load(f)
                        torch.compiler.load_cache_artifacts(cache_data)
                    except Exception as e:
                        print(f"Failed to resotore compile cache from {cache_file}: {e}")

            # Experimental features to reduce compilation times, will be on by default in future
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.triton.unique_kernel_names = True
            torch._inductor.config.fx_graph_cache = True
            torch._inductor.config.triton.cudagraph_trees = True
            torch._inductor.config.triton.cudagraph_support_input_mutation = True
            self.h.compile(fullgraph=True, mode="reduce-overhead")
            self.compiled = True

    def save_compile_cache(self):
        if torch.cuda.is_available() and platform.system() != "Windows" or platform.system() == "macOS":
            cache_path = Path.cwd() / "compile_cache"
            if cache_path.exists() is False:
                cache_path.mkdir(parents=True, exist_ok=True)
            else:
                assert cache_path.is_dir()
                cache_file = (
                    cache_path
                    / f"t2s_decoder_{self.n_layer}_{self.hidden_dim}_{self.n_head}_{self.ffn_dim}_{self.phoneme_vocab_size}_{get_machine_id()}_{torch.__version__}.GSV"
                )
                if cache_file.exists():
                    return

                try:
                    cache = torch.compiler.save_cache_artifacts()
                    assert cache
                    cache_data = cache[0]
                    with open(cache_file, "wb") as f:
                        pickle.dump(cache_data, f)
                except Exception as e:
                    print(f"Failed to save compile cache to {cache_file}: {e}")

    def capture(
        self, input_pos: Tensor, x: Tensor, x_dec: Tensor, kv_caches: MutableSequence[KVCacheProtocol], *args, **kwds
    ) -> CUDAGraph:
        assert torch.cuda.is_available()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        graph = torch.cuda.CUDAGraph()

        with torch.cuda.stream(s):
            for _ in range(5):
                self.h(input_pos, x, kv_caches, *args, **kwds)
        torch.cuda.current_stream().wait_stream(s)

        with torch.cuda.graph(graph):
            x_dec.copy_(self.h(input_pos, x, kv_caches, *args, **kwds))
        torch.cuda.synchronize()

        return graph

    @abstractmethod
    def pre_forward(self, session: T2SSession) -> tuple[list[Tensor], dict[str, Tensor]]:
        return list(), dict()

    @abstractmethod
    def post_forward(self, idx: int, session: T2SSession) -> None:
        return

    def quantize(self, mode: Literal["Int8", "FP8", "FP8_E4M3FN"] | None = None) -> None:
        if mode is None:
            return
        if mode not in {"Int8", "FP8", "FP8_E4M3FN"}:
            raise ValueError(f"Unsupported quantization mode: {mode}")
        match mode:
            case "Int8":
                self.bits = 8
                self.group_size = 32
                import torchao

                torchao.quantization.quantize_(self.h, torchao.quantization.Int8WeightOnlyConfig(self.group_size))

            case "FP8":
                self.bits = 8
                import torchao

                torchao.quantization.quantize_(self.h, torchao.quantization.Float8WeightOnlyConfig())

            case "FP8_E4M3FN":
                self.bits = 8
                replace_all_linear_with_fp8(self.h)

            case _:
                raise ValueError(f"Unsupported Quantization Mode for PyTorch: {mode}")


class CUDAGraphCacheABC(ABC):
    def __init__(
        self,
        decoder: T2SDecoderABC,
    ) -> None:
        self.is_applicable: bool

        if torch.cuda.is_available() and self.is_applicable:
            self.device: torch.device = decoder.device
            self.dtype = decoder.bert_proj.bias.dtype

            self.assigned: bool = False

            self.decoder: T2SDecoderABC = decoder
            self.kv_cache: MutableSequence[KVCacheProtocol] = decoder.init_cache(decoder.max_batch_size)
            self.xy_pos = torch.rand(size=(decoder.max_batch_size, 1, decoder.embedding_dim), device=self.device).to(
                self.dtype
            )
            self.xy_dec = self.xy_pos.clone()

            self.input_pos = torch.tensor([10] * decoder.max_batch_size, device=self.device).int()
            self.graph: torch.cuda.CUDAGraph | None = None
            self.stream: torch.cuda.Stream | None

            self.id: int = random.randint(1, 2**32 - 1)

    def assign_graph(self, session: T2SSession):
        if self.graph is None:
            args, kwds = self.decoder.pre_forward(session)
            graph = self.decoder.capture(self.input_pos, self.xy_pos, self.xy_dec, self.kv_cache, *args, **kwds)
            self.graph = graph
            self.stream = torch.cuda.Stream()

        if self.assigned is False:
            self.get_cache_graph(session)
            session.id = self.id
            self.assigned = True
        else:
            self.capture_new_graph(session)

    @abstractmethod
    def release_graph(self, session: T2SSession): ...

    @abstractmethod
    def get_cache_graph(self, session: T2SSession):
        pass

    @abstractmethod
    def capture_new_graph(self, session: T2SSession):
        pass


class TorchProfiler:
    def __init__(self, debug: bool, log_dir: str = "./profiler/torch") -> None:
        self.debug = debug and os.environ.get("TORCH_PROFILER") == "1"
        self.log_dir = log_dir + "/" + str(time.time())
        self.__profiler: torch.profiler.profile

        if self.debug and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.tensorboard_handler = tensorboard_trace_handler(self.log_dir)

    def profiler_callback(self, prof: torch.profiler.profile):
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        self.tensorboard_handler(prof)

    @staticmethod
    def three_step_schedule(step: int) -> ProfilerAction:
        if step == 0:
            return ProfilerAction.NONE
        elif step == 1:
            return ProfilerAction.RECORD
        elif step == 2:
            return ProfilerAction.RECORD_AND_SAVE
        else:
            return ProfilerAction.RECORD_AND_SAVE

    def start(self):
        if not self.debug:
            return
        assert self.__profiler is not None
        self.__profiler.step()

    def end(self):
        if not self.debug:
            return
        assert self.__profiler is not None
        self.__profiler.step()

    def profiler(self):
        if self.debug:
            activities_list = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities_list.append(torch.profiler.ProfilerActivity.CUDA)

            self.__profiler = torch.profiler.profile(
                activities=activities_list,
                record_shapes=True,
                with_stack=True,
                with_modules=True,
                with_flops=True,
                profile_memory=True,
                schedule=self.three_step_schedule,
                on_trace_ready=self.profiler_callback,
                execution_trace_observer=(
                    ExecutionTraceObserver().register_callback(f"{self.log_dir}/execution_trace.json")
                ),
            )
            return self.__profiler
        else:
            return nullcontext()

    def record(self, func_name: str):
        if self.debug:
            return torch.profiler.record_function(func_name)
        else:
            return nullcontext()
