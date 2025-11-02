from typing import Callable, Protocol, TypeVar, cast

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
def apply_repetition_penalty(logits: Tensor, previous_tokens: Tensor, repetition_penalty: float):
    previous_tokens = previous_tokens.long()
    score = torch.gather(logits, dim=1, index=previous_tokens)
    score = torch.where(
        score < 0,
        score * repetition_penalty,
        score / repetition_penalty,
    )
    logits.scatter_(dim=1, index=previous_tokens, src=score)
    return logits


@script
def apply_greedy_sampling(logits: Tensor):
    return torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.int32)


@script
def apply_temperature(logits: Tensor, temperature: float):
    return logits / temperature


@script
def apply_top_k(logits: Tensor, top_k: int):
    v, _ = torch.topk(logits, top_k)
    pivot = v[:, -1].unsqueeze(-1)
    logits = torch.where(logits < pivot, -float("Inf"), logits)
    return logits


@script
def apply_top_p(logits: Tensor, top_p: float):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    cum_probs[cum_probs > 1] = 1
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[:, 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    return logits


@script
def apply_sampling(logits: Tensor):
    probs = F.softmax(logits, dim=-1)
    q = -torch.log(torch.rand_like(probs))
    idx_next = torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int32)
    return idx_next


class SampleProtocol(Protocol):
    @staticmethod
    def __call__(
        logits: Tensor,
        previous_tokens: Tensor,
        repetition_penalty: float,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tensor: ...


class sample_naive(SampleProtocol):
    @staticmethod
    def __call__(
        logits: Tensor,
        previous_tokens: Tensor,
        repetition_penalty: float = 1.35,
        temperature: float = 1.0,
        top_k: int = 15,
        top_p: float = 1.0,
    ):
        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, previous_tokens, repetition_penalty)

        if temperature <= 1e-5:
            return apply_greedy_sampling(logits)
        elif temperature < 1.0:
            logits = apply_temperature(logits, temperature)

        if top_k < 1025:
            logits = apply_top_k(logits, top_k)

        if top_p < 1.0:
            logits = apply_top_p(logits, top_p)

        return apply_sampling(logits)
