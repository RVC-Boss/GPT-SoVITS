"""
Modified From https://github.com/XXXXRT666/GPT-SoVITS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, MutableSequence, Optional, Protocol

import torch

from .sample_funcs import SampleProtocol, sample_naive

Tensor = torch.Tensor


@dataclass
class T2SResult:
    result: list[Tensor] | None = None
    infer_speed: tuple[float, float] = (0.0, 0.0)
    status: Literal["Success", "Error"] = "Success"
    exception: Optional[Exception] = None
    traceback: Optional[str] = None


@dataclass
class T2SRequest:
    x: list[torch.Tensor]
    x_lens: Tensor
    prompts: torch.Tensor
    bert_feature: list[Tensor]
    valid_length: int
    top_k: int = 5
    top_p: float = 1
    early_stop_num: int = -1
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    use_cuda_graph: bool = False
    debug: bool = False


class KVCacheProtocol(Protocol):
    k_cache: Tensor
    v_cache: Tensor

    def __init__(self, batch_size: int, max_seq_length: int, n_heads: int, head_dim: int) -> None: ...

    def empty(self) -> None: ...

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor, *args, **kwds) -> tuple[Tensor, Tensor]: ...

    def prefill_kv(self, k_val: Tensor, v_val: Tensor) -> None: ...

    def sync_cache(self, kv_cache: KVCacheProtocol) -> None: ...


class T2SDecoderProtocol(Protocol):
    max_seq_length: int
    EOS: int
    n_head: int

    @property
    def device(self) -> torch.device: ...

    def embed(self, x: list[Tensor], y: Tensor, bert_features: list[Tensor]) -> Tensor: ...


class T2SEngineProtocol(Protocol):
    def _handle_request(self, request: T2SRequest) -> tuple[list[Tensor], float, float]: ...

    def generate(self, request: T2SRequest) -> T2SResult: ...


class T2SSession:
    def __init__(
        self,
        decoder: T2SDecoderProtocol,
        request: T2SRequest,
        sapmle_func: type[SampleProtocol] = sample_naive,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        with device:
            self.decoder = decoder
            self.request = request
            self.device = device
            self.dtype = dtype

            bsz = len(request.x)
            y_len = request.prompts.size(-1)
            self.bsz = bsz
            self.y_len = y_len
            request.prompts = request.prompts.to(device, torch.int32)

            # Cache
            self.kv_cache: MutableSequence[KVCacheProtocol]
            self.sample = sapmle_func()

            # Forward args
            self.x = [i.to(device) for i in request.x]
            self.x_lens = request.x_lens.to(torch.int32)
            self.y = torch.zeros((bsz, decoder.max_seq_length)).to(torch.int32)
            self.y[:, : request.prompts.shape[-1]] = request.prompts
            self.bert_feature = [i.to(device, dtype) for i in request.bert_feature]

            self.prefill_len = self.x_lens + request.prompts.size(1)

            self.input_pos = torch.zeros_like(self.prefill_len)
            self.input_pos.add_(self.prefill_len)

            # CUDA Graph
            self.stream: Optional[torch.cuda.Stream] = None
            self.graph: Optional[torch.cuda.CUDAGraph] = None
            self.xy_pos_: Tensor
            self.xy_dec_: Tensor

            # EOS
            self.completed = torch.Tensor([False] * len(self.x)).bool().to(device)
            self.y_results: list[Tensor] = [None] * len(self.x)  # type: ignore

            self.xy_pos = decoder.embed(self.x, request.prompts, self.bert_feature)

            max_len = int(self.prefill_len.max().item())
            attn_mask = torch.zeros(size=(bsz, max_len, max_len), dtype=torch.bool)

            for bs in range(bsz):
                pos = int(self.x_lens[bs])
                seq_len = pos + y_len

                attn_mask[bs, :seq_len, :pos] = True

                ar_mask = ~torch.triu(
                    input=torch.ones(
                        size=(
                            y_len,
                            y_len,
                        ),
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )
                attn_mask[bs, pos:seq_len, pos:seq_len] = ar_mask

            self.attn_mask = attn_mask
            self.attn_mask = attn_mask.unsqueeze(0).expand(-1, decoder.n_head, -1, -1)

            self.id: int = -1

            # Sage Attn & Transformer Engine Impl
            self.cu_seqlens_q: Tensor
            self.cu_seqlens_kv: Tensor
