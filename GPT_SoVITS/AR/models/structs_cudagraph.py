from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import torch

Tensor = torch.Tensor


@dataclass
class T2SResult:
    result: List[Tensor] | None = None
    infer_speed: float = 0.0
    status: Literal["Success", "Error"] = "Success"
    exception: Optional[Exception] = None
    traceback: Optional[str] = None


@dataclass
class T2SRequest:
    x: List[torch.Tensor]
    x_lens: Tensor
    prompts: torch.Tensor
    bert_feature: List[Tensor]
    valid_length: int
    top_k: int = 5
    top_p: float = 1
    early_stop_num: int = -1
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    use_cuda_graph: bool = False
    debug: bool = False


class T2SSession:
    def __init__(self, decoder, request: T2SRequest, device: torch.device, dtype: torch.dtype):
        with device:
            self.decoder = decoder
            self.request = request
            self.device = device
            self.dtype = dtype

            bsz = len(request.x)
            y_len = request.prompts.size(-1)
            self.bsz = bsz
            self.y_len = y_len

            from AR.models.t2s_model_cudagraph import Sampler

            self.sampler = Sampler(bsz, decoder.vocab_size)

            self.x = request.x
            self.x_lens = request.x_lens.to(torch.int32)
            self.y = request.prompts
            self.bert_feature = request.bert_feature

            self.prefill_len = self.x_lens + self.y.size(1)

            self.input_pos = torch.zeros_like(self.prefill_len)
            self.input_pos.add_(self.prefill_len)

            self.completed = torch.Tensor([False] * len(self.x)).bool().to(device)
            self.y_results: List[Tensor] = [None] * len(self.x)  # type: ignore

            self.xy_pos = decoder.embed(self.x, self.y, self.bert_feature)

            attn_mask = []
            for bs in range(bsz):
                pos = int(self.x_lens[bs].item())
                mask = torch.zeros(pos + y_len, pos + y_len).bool()
                mask[:, :pos].fill_(True)
                if y_len > 0:
                    mask[-y_len:, -y_len:] = ~torch.triu(
                        torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                    )
                attn_mask.append(mask)
            self.attn_mask_nested = torch.nested.nested_tensor(attn_mask)
