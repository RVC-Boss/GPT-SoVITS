from typing import Protocol

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


class SampleProtocol(Protocol):
    @staticmethod
    def __call__(
        logits: Tensor,
        previous_tokens: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> Tensor: ...


class sample_naive(SampleProtocol):
    @staticmethod
    def __call__(
        logits: Tensor,
        previous_tokens: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ):
        if temperature <= 1e-5:
            probs = F.softmax(logits, dim=-1)
            return torch.argmax(probs, dim=-1, keepdim=True).to(dtype=torch.int32)

        if repetition_penalty != 1.0:
            previous_tokens = previous_tokens.long()
            score = torch.gather(logits, dim=1, index=previous_tokens)
            score = torch.where(
                score < 0,
                score * repetition_penalty,
                score / repetition_penalty,
            )
            logits.scatter_(dim=1, index=previous_tokens, src=score)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cum_probs[cum_probs > 1] = 1
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[:, 0] = False  # keep at least one option
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        if temperature < 1.0:
            logits /= temperature

        v, _ = torch.topk(logits, top_k)
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

        probs = F.softmax(logits, dim=-1)
        q = -torch.log(torch.rand_like(probs))
        idx_next = torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int32)

        return idx_next
