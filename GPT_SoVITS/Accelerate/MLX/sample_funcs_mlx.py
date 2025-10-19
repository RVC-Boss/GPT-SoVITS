from functools import partial
from typing import Protocol

import mlx.core as mx

Array = mx.array


class SampleProtocolMLX(Protocol):
    @staticmethod
    def __call__(
        logits: Array,
        previous_tokens: Array,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> Array: ...


def apply_repetition_penalty(logits: Array, previous_tokens: Array, repetition_penalty: float):
    batch_idx = mx.arange(previous_tokens.shape[0])
    selected_logits = logits[batch_idx, previous_tokens]
    selected_logits = mx.where(
        selected_logits < 0, selected_logits * repetition_penalty, selected_logits / repetition_penalty
    )
    logits[batch_idx, previous_tokens] = selected_logits
    return logits


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_greedy_sampling(logits: Array):
    return mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_temperature(logits: Array, temperature: float):
    return logits / temperature


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_k(logits: Array, top_k: int):
    v = mx.topk(logits, top_k)
    pivot = mx.expand_dims(v[:, 0], -1)
    logits = mx.where(logits < pivot, -mx.inf, logits)
    return logits


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_p(logits: Array, top_p: float):
    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    cum_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[:, -1] = False
    indices_to_remove = mx.zeros_like(logits).astype(mx.bool_)
    batch_indices = mx.arange(logits.shape[0])[:, None]
    indices_to_remove[batch_indices, sorted_indices] = sorted_indices_to_remove
    logits = mx.where(indices_to_remove, -mx.inf, logits)
    return logits


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_sampling(logits: Array):
    gumbel_noise = mx.random.gumbel(shape=logits.shape, dtype=logits.dtype)
    idx_next = mx.argmax(logits + gumbel_noise, axis=-1, keepdims=True).astype(mx.int32)
    return idx_next


class sample_naive(SampleProtocolMLX):
    @staticmethod
    def __call__(
        logits,
        previous_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
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
