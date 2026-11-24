import torch
from torch.nn.functional import *
from torch.nn.functional import (
    _canonical_mask,
)


def _reshape_qkv_for_attention(proj_qkv, query):
    """Split packed QKV and move its selector axis first without aten::unflatten."""
    return proj_qkv.reshape(
        query.size(0), query.size(1), 3, query.size(2)
    ).permute(2, 0, 1, 3).contiguous()


def _scaled_dot_product_attention_onnx(q, k, v, attn_mask, is_causal=False):
    """ONNX-exportable dropout-free attention for canonical additive masks."""
    if is_causal:
        raise ValueError(
            "ONNX cached attention requires an explicit mask and is_causal=False."
        )
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q * scale, k.transpose(-2, -1))
    if attn_mask is not None:
        scores = scores + attn_mask
    probabilities = softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def multi_head_attention_forward_patched(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    average_attn_weights=True,
    is_causal=False,
    cache=None,
):
    # set up shape vars
    _, _, embed_dim = query.shape
    attn_mask = _canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query.dtype,
        check_other=False,
    )
    head_dim = embed_dim // num_heads

    proj_qkv = linear(query, in_proj_weight, in_proj_bias)
    proj_qkv = _reshape_qkv_for_attention(proj_qkv, query)
    q, k, v = proj_qkv[0], proj_qkv[1], proj_qkv[2]

    if cache["first_infer"] == 1:
        cache["k"][cache["stage"]] = k
        cache["v"][cache["stage"]] = v
    else:
        cache["k"][cache["stage"]] = torch.cat([cache["k"][cache["stage"]][:-1], k], 0)
        cache["v"][cache["stage"]] = torch.cat([cache["v"][cache["stage"]][:-1], v], 0)
        k = cache["k"][cache["stage"]]
        v = cache["v"][cache["stage"]]
    cache["stage"] = (cache["stage"] + 1) % cache["all_stage"]

    attn_mask = _canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )
    attn_mask = attn_mask.unsqueeze(0)

    q = q.view(-1, num_heads, head_dim).transpose(0, 1)
    k = k.view(-1, num_heads, head_dim).transpose(0, 1)
    v = v.view(-1, num_heads, head_dim).transpose(0, 1)

    dropout_p = 0.0
    attn_mask = attn_mask.unsqueeze(0)
    q = q.view(num_heads, -1, head_dim).unsqueeze(0)
    k = k.view(num_heads, -1, head_dim).unsqueeze(0)
    v = v.view(num_heads, -1, head_dim).unsqueeze(0)
    attn_output = _scaled_dot_product_attention_onnx(
        q, k, v, attn_mask, is_causal=is_causal
    )
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(-1, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(-1, 1, attn_output.size(1))

    return attn_output
