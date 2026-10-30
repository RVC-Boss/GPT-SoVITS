# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/activation.py
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter

from AR.modules.patched_mha_with_cache import multi_head_attention_forward_patched


class MultiheadAttention(Module):
    """Inference attention with the original packed-projection checkpoint keys."""

    __constants__ = ["batch_first"]

    def __init__(self, embed_dim, num_heads, batch_first=False, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, **factory_kwargs)
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        cache=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError("only bool and floating types of key_padding_mask are supported")

        tensor_args = (query, key, value, self.in_proj_weight, self.in_proj_bias,
                       self.out_proj.weight, self.out_proj.bias)
        native_attention = (
            is_batched and self.batch_first and query is key and key is value
            and cache is None and attn_mask is None
            and query.dtype == self.in_proj_bias.dtype == self.in_proj_weight.dtype
            and not (query.is_nested and key_padding_mask is not None)
            and self.num_heads % 2 == 0 and not torch.is_autocast_enabled()
            and not torch.overrides.has_torch_function(tensor_args)
            and all(t.device.type in {"cpu", "cuda"} for t in tensor_args)
            and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensor_args))
        )
        if native_attention:
            return torch._native_multi_head_attention(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias, self.out_proj.weight, self.out_proj.bias,
                key_padding_mask, need_weights, average_attn_weights,
                1 if key_padding_mask is not None else None,
            )

        assert not (query.is_nested or key.is_nested or value.is_nested), (
            "NestedTensor requires native attention without an explicit attention mask or KV cache"
        )
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = query.transpose(1, 0), key.transpose(1, 0)
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        output, weights = multi_head_attention_forward_patched(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias, self.out_proj.weight, self.out_proj.bias,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights, cache=cache,
        )
        return (output.transpose(1, 0) if self.batch_first and is_batched else output), weights
