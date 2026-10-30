"""Packed-projection inference attention with per-layer KV caching."""

import math

import torch
from torch.nn import functional as F


def multi_head_attention_forward_patched(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    cache=None,
):
    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)
    if not is_batched:
        query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask, mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask), other_name="attn_mask", target_type=query.dtype,
    )
    attn_mask = F._canonical_mask(
        mask=attn_mask, mask_name="attn_mask", other_type=None,
        other_name="", target_type=query.dtype, check_other=False,
    )
    assert embed_dim == embed_dim_to_check, "unexpected attention embedding dimension"
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    assert key.shape == value.shape, "key and value shapes must match"
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    if cache is not None:
        stage = cache["stage"]
        if cache["first_infer"] == 1:
            cache["k"][stage] = k
            cache["v"][stage] = v
        else:
            k = cache["k"][stage] = torch.cat([cache["k"][stage], k], dim=0)
            v = cache["v"][stage] = torch.cat([cache["v"][stage], v], dim=0)
            src_len = k.shape[0]
        cache["stage"] = (stage + 1) % cache["all_stage"]

    attn_mask = F._canonical_mask(
        mask=attn_mask, mask_name="attn_mask", other_type=None,
        other_name="", target_type=q.dtype, check_other=False,
    )
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            expected = (tgt_len, src_len)
            if attn_mask.shape != expected:
                raise RuntimeError(f"The 2D attn_mask shape is {attn_mask.shape}, expected {expected}")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            expected = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != expected:
                raise RuntimeError(f"The 3D attn_mask shape is {attn_mask.shape}, expected {expected}")
        else:
            raise RuntimeError(f"attn_mask dimension {attn_mask.dim()} is not supported")

    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), "unexpected key padding mask shape"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        )
        attn_mask = key_padding_mask if attn_mask is None else attn_mask + key_padding_mask

    if need_weights:
        q_scaled = q / math.sqrt(q.size(-1))
        weights = (torch.bmm(q_scaled, k.transpose(-2, -1)) if attn_mask is None
                   else torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1)))
        weights = F.softmax(weights, dim=-1)
        output = torch.bmm(weights, v)
        output = output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        output = F.linear(output, out_proj_weight, out_proj_bias).view(tgt_len, bsz, embed_dim)
        weights = weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            weights = weights.mean(dim=1)
        if not is_batched:
            output, weights = output.squeeze(1), weights.squeeze(0)
        return output, weights

    if attn_mask is not None:
        if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(0)
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)
    q = q.view(bsz, num_heads, tgt_len, head_dim)
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)
    output = F.scaled_dot_product_attention(q, k, v, attn_mask, 0.0, False)
    output = output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
    output = F.linear(output, out_proj_weight, out_proj_bias).view(tgt_len, bsz, embed_dim)
    return (output.squeeze(1) if not is_batched else output), None
