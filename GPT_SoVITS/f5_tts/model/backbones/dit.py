"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from x_transformers.x_transformers import RotaryEmbedding

from GPT_SoVITS.f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)

from module.commons import sequence_mask


class TextEmbedding(nn.Module):
    def __init__(self, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        batch, text_len = text.shape[0], text.shape[1]

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]

            # print(23333333,text.shape,text_pos_embed.shape)#torch.Size([7, 465, 256]) torch.Size([7, 465, 256])

            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        use_step_embedding=True,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.d_embed = TimestepEmbedding(dim) if use_step_embedding else None
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def prepare_static_cache(self, cond0, x_lens, text0):
        text = text0.transpose(2, 1)
        cond = cond0.transpose(2, 1)
        seq_len = cond.shape[1]
        text_embed = self.text_embed(text, seq_len, drop_text=False)
        projection = self.input_embed.proj
        mel_dim = cond.shape[-1]
        static = F.linear(
            torch.cat((cond, text_embed), dim=-1),
            projection.weight[:, mel_dim:], projection.bias,
        )
        negative_static = F.linear(
            torch.cat((torch.zeros_like(cond), text_embed), dim=-1),
            projection.weight[:, mel_dim:], projection.bias,
        )
        return {
            "condition": static,
            "negative_condition": negative_static,
            "mask": sequence_mask(x_lens, max_length=seq_len).to(cond.device),
            "rope": self.rotary_embed.forward_from_seq_len(seq_len),
        }

    def forward(  # x, prompt_x, x_lens, t, style,cond
        self,  # d is channel,n is T
        x0: float["b n d"],  # nosied input audio  # noqa: F722
        cond0: float["b n d"],  # masked cond audio  # noqa: F722
        x_lens,
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        dt_base_bootstrap=None,
        text0=None,  # : int["b nt"]  # noqa: F722#####condition feature
        use_grad_ckpt=False,  # bool
        ###no-use
        drop_audio_cond=False,  # cfg for cond audio
        drop_text=False,  # cfg for text
        # mask: bool["b n"] | None = None,  # noqa: F722
        infer=False,  # bool
        text_cache=None,  # torch tensor as text_embed
        dt_cache=None,  # torch tensor as dt
        static_cache=None,
        # cache-dit / DBCache (see module/cache_dit.py). `dbcache=None` => untouched path.
        dbcache=None,  # module.cache_dit.DBCacheState
        dbcache_branch=0,  # 0 = CFG positive, 1 = CFG negative
    ):
        x, t, mask, rope, text_embed, dt = self.trunk(
            x0, cond0, x_lens, time, dt_base_bootstrap=dt_base_bootstrap, text0=text0,
            drop_audio_cond=drop_audio_cond, drop_text=drop_text, infer=infer,
            text_cache=text_cache, dt_cache=dt_cache, static_cache=static_cache,
        )

        residual = x
        if self.long_skip_connection is None:
            residual = None

        if dbcache is None or not dbcache.enabled:
            # ---- original path: all blocks, then head ------------------------------
            x = self.run_blocks(x, t, mask, rope, 0, self.depth, use_grad_ckpt)
        else:
            # ---- cache-dit: front -> (hit: head only | miss: mid -> head) -----------
            fn = max(0, min(int(dbcache.fn), self.depth - 1))
            x_f = self.run_blocks(x, t, mask, rope, 0, fn, use_grad_ckpt)
            fd = dbcache.front_update(dbcache_branch, x_f, x)
            if dbcache.can_cache(dbcache_branch, fd):
                x = x_f + dbcache.mid_residual(dbcache_branch)
                dbcache.on_hit(dbcache_branch)
                if dbcache.trace:
                    print("[dbcache] step=%d HIT  fd=%.5f (hits=%d misses=%d)"
                          % (dbcache.step, fd, dbcache.branch(dbcache_branch).hits,
                             dbcache.branch(dbcache_branch).misses), flush=True)
            else:
                x = self.run_blocks(x_f, t, mask, rope, fn, self.depth, use_grad_ckpt)
                dbcache.on_miss(dbcache_branch, x, x_f)
                if dbcache.trace:
                    print("[dbcache] step=%d MISS fd=%.5f (hits=%d misses=%d)"
                          % (dbcache.step, fd, dbcache.branch(dbcache_branch).hits,
                             dbcache.branch(dbcache_branch).misses), flush=True)

        output = self.head(x, t, residual)

        if infer:
            return output, text_embed if static_cache is None else None, dt
        else:
            return output

    # ---------------------------------------------------------------------------------
    # Three-segment split used by cache-dit; the weights are shared, nothing is copied.
    # ---------------------------------------------------------------------------------
    def trunk(self, x0, cond0, x_lens, time, dt_base_bootstrap=None, text0=None,
              drop_audio_cond=False, drop_text=False, infer=False, text_cache=None,
              dt_cache=None, static_cache=None):
        """Everything before the transformer blocks: t/dt embedding + input embedding."""
        text_embed = None
        x = x0.transpose(2, 1)
        cond = cond0.transpose(2, 1)
        text = text0.transpose(2, 1)
        mask = (static_cache["mask"] if static_cache is not None
                else sequence_mask(x_lens, max_length=x.size(1)).to(x.device))

        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        if self.d_embed is None:
            dt = None
        elif infer and dt_cache is not None:
            dt = dt_cache
        else:
            dt = self.d_embed(dt_base_bootstrap)
        if dt is not None:
            t += dt

        if static_cache is not None:
            static = static_cache["negative_condition" if drop_audio_cond else "condition"]
            x = F.linear(x, self.input_embed.proj.weight[:, :x.shape[-1]]) + static
            x = self.input_embed.conv_pos_embed(x, mask=mask) + x
        elif infer and text_cache is not None:
            text_embed = text_cache
            x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text)  ###need to change
            x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = (static_cache["rope"] if static_cache is not None
                else self.rotary_embed.forward_from_seq_len(seq_len))

        return x, t, mask, rope, text_embed, dt

    def run_blocks(self, x, t, mask, rope, start=0, end=None, use_grad_ckpt=False):
        """`front` is blocks[0:fn], `mid` is blocks[fn:depth]."""
        end = self.depth if end is None else end
        for block in self.transformer_blocks[start:end]:
            if use_grad_ckpt:
                x = checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)
        return x

    def head(self, x, t, residual=None):
        """`norm_out -> proj_out`, computed on every step, hits included."""
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = self.norm_out(x, t)
        return self.proj_out(x)

