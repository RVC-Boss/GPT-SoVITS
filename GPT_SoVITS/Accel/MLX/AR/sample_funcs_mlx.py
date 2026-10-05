from functools import partial
from typing import Protocol

import mlx.core as mx


Array = mx.array


class SampleProtocolMLX(Protocol):
    @staticmethod
    def __call__(
        logits: Array,
        valid_lens: Array,
        previous_tokens: Array,
        repetition_penalty: Array,
        temperature: Array,
        top_k: Array,
        top_p: Array,
    ) -> Array: ...


def apply_repetition_penalty(logits: Array, previous_tokens: Array, valid_lens: Array, repetition_penalty: Array):
    _, T = previous_tokens.shape

    arange_t = mx.arange(T).reshape(1, -1)
    valid_mask = arange_t < valid_lens.reshape(-1, 1)

    safe_tokens = mx.where(
        valid_mask,
        previous_tokens,
        mx.zeros_like(previous_tokens),
    )  # [B, T]

    # [B, T]
    score = mx.take_along_axis(logits, safe_tokens, axis=1)

    rp = repetition_penalty.reshape(-1, 1)

    score = mx.where(
        score < 0,
        score * rp,
        score / rp,
    )

    score = mx.where(
        valid_mask,
        score,
        mx.take_along_axis(logits, safe_tokens, axis=1),
    )

    logits = mx.put_along_axis(logits, safe_tokens, score, axis=1)

    return logits


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_greedy_sampling(logits: Array):
    return mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_temperature(logits: Array, temperature: Array, temp_mask: Array):
    temp = temperature[:]
    temp[temp_mask] = 1
    return logits / temp


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_k(logits: Array, top_k: Array):
    sorted_logits = -mx.sort(-logits)
    pivot = mx.take_along_axis(
        sorted_logits,
        (top_k - 1).reshape(-1, 1),
        axis=1,
    )
    logits = mx.where(logits < pivot, -mx.inf, logits)
    return logits


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_p(logits: Array, top_p: Array):
    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    cum_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[:, -1] = False
    indices_to_remove = mx.zeros_like(logits).astype(mx.bool_)
    indices_to_remove = mx.put_along_axis(indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=1)
    logits = mx.where(indices_to_remove, -mx.inf, logits)
    return logits


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_sampling(logits: Array, greedy_mask: Array):
    gumbel_noise = mx.random.gumbel(shape=logits.shape, dtype=logits.dtype)
    idx_next = mx.argmax(logits + gumbel_noise, axis=-1, keepdims=True).astype(mx.int32)

    if mx.any(greedy_mask):
        idx_greedy = mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)
        idx_next = mx.where(greedy_mask.reshape(-1, 1), idx_greedy, idx_next)
    return idx_next


class sample_naive_mlx(SampleProtocolMLX):
    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    @staticmethod
    def __call__(
        logits: Array,
        valid_lens: Array,
        previous_tokens: Array,
        repetition_penalty: Array,
        temperature: Array,
        top_k: Array,
        top_p: Array,
    ) -> Array:
        if mx.any(mx.not_equal(repetition_penalty, 1.0)):
            logits = apply_repetition_penalty(logits, previous_tokens, valid_lens, repetition_penalty)

        greedy_mask = temperature <= 1e-5
        temp_mask = (temperature < 1.0) & (~greedy_mask)

        if mx.any(temp_mask):
            logits = apply_temperature(logits, temperature, temp_mask)

        if mx.any(top_k < 1025):
            logits = apply_top_k(logits, top_k)

        if mx.any(top_p < 1.0):
            logits = apply_top_p(logits, top_p)

        return apply_sampling(logits, greedy_mask)
