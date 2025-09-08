from typing import Protocol, cast

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


class sample_naive(SampleProtocolMLX):
    # @partial(mx.compile)
    @staticmethod
    def __call__(
        logits,
        previous_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    ):
        if temperature <= 1e-5:
            probs = mx.softmax(logits, axis=-1)
            return mx.argmax(probs, axis=-1, keepdims=True).astype(mx.int32)

        if repetition_penalty != 1.0:
            batch_idx = mx.arange(cast(tuple[int, ...], previous_tokens.shape)[0])
            previous_tokens = previous_tokens.astype(mx.int64)
            selected_logists = logits[batch_idx, previous_tokens]
            selected_logists = mx.where(
                selected_logists < 0, selected_logists * repetition_penalty, selected_logists / repetition_penalty
            )
            logits[batch_idx, previous_tokens] = selected_logists

        if top_p < 1.0:
            sorted_indices = mx.argsort(-logits, axis=-1)
            sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
            cum_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[:, -1] = False
            indices_to_remove = mx.zeros_like(logits).astype(mx.bool_)
            batch_indices = mx.arange(cast(tuple[int, ...], logits.shape)[0])[:, None]
            indices_to_remove[batch_indices, sorted_indices] = sorted_indices_to_remove
            logits = mx.where(indices_to_remove, -mx.inf, logits)

        if temperature < 1.0:
            logits = logits / temperature

        v = mx.topk(logits, top_k)
        pivot = mx.expand_dims(v[:, 0], -1)
        logits = mx.where(logits < pivot, -mx.inf, logits)

        gumbel_noise = mx.random.gumbel(shape=cast(tuple[int, ...], logits.shape), dtype=logits.dtype)
        idx_next = mx.argmax(logits + gumbel_noise, axis=-1, keepdims=True).astype(mx.int32)

        return idx_next
