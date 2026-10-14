"""
Modified From https://github.com/XXXXRT666/GPT-SoVITS
"""

from __future__ import annotations

from collections.abc import Generator, MutableSequence
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Literal, Optional, Protocol, Union

import torch
from rich.progress import Progress, TaskID


if TYPE_CHECKING:
    from .t2s_model_abc import T2SDecoderABC


Tensor = torch.Tensor
device = torch.device
dtype = torch.dtype


@dataclass
class T2SResult:
    result: Optional[list[Tensor]] = None
    infer_speed: tuple[float, float] = (0.0, 0.0)  # Speed, Time
    total_tokens: int = 0
    status: Literal["Success", "Error"] = "Success"
    request_id: Optional[Union[str, int]] = None
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
    request_id: Union[str, int] = -1
    stream_interval: Optional[int] = None
    return_partial: bool = True
    debug: bool = False


@dataclass
class T2SStreamResponse:
    request_id: Union[str, int]
    step: int
    tokens: Optional[list[Tensor]]
    finished: bool = False
    result: Optional[T2SResult] = None
    exception: Optional[Exception] = None


@dataclass
class T2SRequestHandle:
    request_id: Union[str, int]
    queue: SimpleQueue[T2SStreamResponse] = field(default_factory=SimpleQueue)
    done: bool = False

    def get_nowait(self) -> Optional[T2SStreamResponse]:
        try:
            return self.queue.get_nowait()
        except Empty:
            return None

    async def aget_nowait(self) -> Optional[T2SStreamResponse]:
        # Non-blocking coroutine wrapper for async callers.
        return self.get_nowait()


KVCache = tuple[Tensor, Tensor]


class KVCacheProtocol(Protocol):
    @staticmethod
    def empty(kv_cache: KVCache) -> None: ...

    @staticmethod
    def update(input_pos: Tensor, k_val: Tensor, v_val: Tensor, kv_cache: KVCache) -> KVCache: ...

    @staticmethod
    def prefill_kv(k_val: Tensor, v_val: Tensor, kv_cache: KVCache) -> None: ...

    @staticmethod
    def init_cache(
        batch_size: int, max_seq_length: int, n_heads: int, head_dim: int, device: device, dtype: dtype
    ) -> KVCache: ...

    @staticmethod
    def sync_cache(tgt: KVCache, src: KVCache) -> None: ...


class T2SEngineProtocol(Protocol):
    def generate(self, request: T2SRequest, show_progress: bool = False) -> T2SResult: ...

    def stream_generate(
        self, request: T2SRequest, show_progress: bool = False
    ) -> Generator[T2SStreamResponse, None, None]: ...

    def shutdown(self) -> None: ...


cpu = torch.device("cpu")


class T2SSession:
    def __init__(
        self,
        decoder: T2SDecoderABC,
        request: T2SRequest,
        device: torch.device = cpu,
        dtype: torch.dtype = torch.float32,
    ):
        with device:
            self.decoder = decoder
            self.request = request
            self.device = device
            self.dtype = dtype

            bsz = len(request.x)
            prompt_len = request.prompts.size(-1)
            self.bsz = bsz
            self.prompt_len = prompt_len
            self.step_count = 0
            self.request_id = request.request_id or id(self)
            self.stream_interval = request.stream_interval if request.stream_interval is not None else 0
            self.last_stream_step = 0
            request.prompts = request.prompts.to(device, torch.int32)

            # Cache in prefill
            self.kv_cache: MutableSequence[KVCache]

            # Forward args
            self.x = [i.to(device) for i in request.x]
            self.x_lens = request.x_lens.to(device, torch.int32)
            self.y = torch.zeros((bsz, decoder.max_seq_length), device=device, dtype=torch.int32)
            self.y[:, : request.prompts.shape[-1]] = request.prompts
            self.bert_feature = [i.to(device, dtype) for i in request.bert_feature]

            self.prefill_len = self.x_lens + request.prompts.size(1)

            self.input_pos = self.prefill_len.clone().reshape(bsz)
            self.max_decode_steps = min(int(decoder.max_seq_length - int(self.input_pos.max().item())), 640)
            self.max_decode_steps = max(1, self.max_decode_steps)

            # EOS
            self.completed = torch.zeros(bsz, device=device, dtype=torch.bool)
            self.output_lengths = [0] * bsz
            self.y_results: list[Tensor] = [None] * len(self.x)  # type: ignore

            max_len = int(self.prefill_len.max().item())
            attn_mask = torch.zeros(size=(bsz, max_len, max_len), dtype=torch.bool)

            for bs in range(bsz):
                pos = int(self.x_lens[bs])
                seq_len = pos + prompt_len

                attn_mask[bs, :seq_len, :pos] = True

                ar_mask = ~torch.triu(
                    input=torch.ones(
                        size=(
                            prompt_len,
                            prompt_len,
                        ),
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )
                attn_mask[bs, pos:seq_len, pos:seq_len] = ar_mask

            self.attn_mask = attn_mask.unsqueeze(1)

            # Streaming / book-keeping
            self.start_time: float = 0.0
            self.total_tokens: int = 0

            self.prefill_hidden: Tensor
            self.progress_task: Optional[TaskID] = None
            self.progress: Optional[Progress] = None

            self.id: int = id(self)
            self.slot_indices: list[int] = []

            # Sage Attn & Transformer Engine Impl
            self.cu_seqlens_q: Tensor
            self.cu_seqlens_kv: Tensor
