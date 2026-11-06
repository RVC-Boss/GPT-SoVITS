from collections.abc import Callable
from typing import Protocol, TypeVar, cast

import torch
import torch.nn.functional as F
from typing_extensions import ParamSpec


P = ParamSpec("P")
R = TypeVar("R")
Tensor = torch.Tensor


def script(fn: Callable[P, R]) -> Callable[P, R]:
    scripted = torch.jit.script(fn)
    return cast(Callable[P, R], scripted)


@script
def apply_repetition_penalty(
    logits: Tensor,
    previous_tokens: Tensor,
    valid_lens: Tensor,
    repetition_penalty: Tensor,
):
    _, T = previous_tokens.shape
    device = logits.device

    previous_tokens = previous_tokens.long()
    valid_lens = valid_lens.long()

    arange_t = torch.arange(T, device=device).unsqueeze(0)
    valid_mask = arange_t < valid_lens.unsqueeze(1)

    safe_tokens = torch.where(
        valid_mask,
        previous_tokens,
        torch.zeros_like(previous_tokens),
    )  # [B, T]

    # [B, T]
    score = torch.gather(logits, dim=1, index=safe_tokens)

    rp = repetition_penalty.to(logits.dtype).unsqueeze(1)

    score = torch.where(
        score < 0,
        score * rp,
        score / rp,
    )

    score = torch.where(
        valid_mask,
        score,
        torch.gather(logits, dim=1, index=safe_tokens),
    )

    logits.scatter_(dim=1, index=safe_tokens, src=score)

    return logits


@script
def apply_greedy_sampling(logits: Tensor):
    return torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.int32)


@script
def apply_temperature(logits: Tensor, temperature: Tensor, temp_mask: Tensor):
    return logits / temperature.to(logits.dtype).unsqueeze(1).clamp_min(1e-5)


@script
def apply_top_k(logits: Tensor, top_k: Tensor):
    sorted_logits, _ = torch.sort(logits, descending=True)
    pivot = torch.gather(
        sorted_logits,
        dim=1,
        index=(top_k.long() - 1).clamp(0, logits.size(1) - 1).unsqueeze(1),
    )
    logits = torch.where(logits < pivot, -float("Inf"), logits)
    return logits


@script
def apply_top_p(logits: Tensor, top_p: Tensor):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    cum_probs[cum_probs > 1] = 1
    sorted_indices_to_remove = cum_probs > top_p.unsqueeze(1)
    sorted_indices_to_remove[:, 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    return logits


@script
def apply_sampling(logits: Tensor, greedy_mask: Tensor):
    probs = F.softmax(logits, dim=-1)
    q = -torch.log(torch.rand_like(probs))
    idx_next = torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int32)

    if torch.any(greedy_mask):
        idx_greedy = torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.int32)
        idx_next = torch.where(greedy_mask.unsqueeze(1), idx_greedy, idx_next)
    return idx_next


class SampleProtocol(Protocol):
    @staticmethod
    def __call__(
        logits: Tensor,
        valid_lens: Tensor,
        previous_tokens: Tensor,
        repetition_penalty: Tensor,
        temperature: Tensor,
        top_k: Tensor,
        top_p: Tensor,
    ) -> Tensor: ...


class sample_naive(SampleProtocol):
    @staticmethod
    def __call__(
        logits: Tensor,
        valid_lens: Tensor,
        previous_tokens: Tensor,
        repetition_penalty: Tensor,
        temperature: Tensor,
        top_k: Tensor,
        top_p: Tensor,
    ):
        if torch.any(repetition_penalty != 1.0):
            logits = apply_repetition_penalty(logits, previous_tokens, valid_lens, repetition_penalty)

        greedy_mask = temperature <= 1e-5
        temp_mask = (~greedy_mask) & (temperature != 1.0)

        if torch.any(temp_mask):
            logits = apply_temperature(logits, temperature, temp_mask)

        if torch.any(top_k < 1025):
            logits = apply_top_k(logits, top_k)

        if torch.any(top_p < 1.0):
            logits = apply_top_p(logits, top_p)

        return apply_sampling(logits, greedy_mask)
