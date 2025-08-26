from torch.nn.functional import *
from torch.nn.functional import (
    _canonical_mask,
)
from typing import Tuple, Optional


def multi_head_attention_forward_patched(
    query,
    key,
    value,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    cache=None,
) -> Tuple[Tensor, Optional[Tensor]]:
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
    proj_qkv = proj_qkv.unflatten(-1, (3, query.size(-1))).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
    q, k, v = proj_qkv[0], proj_qkv[1], proj_qkv[2]

    # 使用动态形状推断来统一处理kv cache首步和后续步骤形状差异
    # # k,v : [N, 1, 512] at first time, [1, 1, 512] afterwards
    # # cache_k, cache_v : [1, N, 1, 512] size increasement is prepared outside
    first_infer_mask = cache["first_infer"]
    cache_k = cache["k"][cache["stage"]]
    cache_v = cache["v"][cache["stage"]]
    # Magic to get an index of either -1 or -N according to if first_infer_mask is set
    minus_one = torch.tensor([-1]).to(k.device).to(torch.int64)
    multipled = minus_one * first_infer_mask * (cache['x_seq_len'] + cache['y_seq_len'])
    index_offset = torch.min(minus_one, multipled)
    cache_k[index_offset :, :, :] = k
    cache_v[index_offset :, :, :] = v
    cache["k"][cache["stage"]] = cache_k
    cache["v"][cache["stage"]] = cache_v
    k = cache_k
    v = cache_v

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
    attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(-1, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(-1, 1, attn_output.size(1))

    return attn_output
