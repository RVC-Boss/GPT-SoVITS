"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers import RMSNorm
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    Attention,
    AttnProcessor,
    FeedForward,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

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
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
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


# Flat UNet Transformer backbone


class UNetT(nn.Module):
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
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        skip_connect_type: Literal["add", "concat", "none"] = "concat",
    ):
        super().__init__()
        assert depth % 2 == 0, "UNet-Transformer's depth should be even."

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        # transformer layers & skip connections

        self.dim = dim
        self.skip_connect_type = skip_connect_type
        needs_skip_proj = skip_connect_type == "concat"

        self.depth = depth
        self.layers = nn.ModuleList([])

        for idx in range(depth):
            is_later_half = idx >= (depth // 2)

            attn_norm = RMSNorm(dim)
            attn = Attention(
                processor=AttnProcessor(),
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            )

            ff_norm = RMSNorm(dim)
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

            skip_proj = nn.Linear(dim * 2, dim, bias=False) if needs_skip_proj and is_later_half else None

            self.layers.append(
                nn.ModuleList(
                    [
                        skip_proj,
                        attn_norm,
                        attn,
                        ff_norm,
                        ff,
                    ]
                )
            )

        self.norm_out = RMSNorm(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        # postfix time t to input x, [b n d] -> [b n+1 d]
        x = torch.cat([t.unsqueeze(1), x], dim=1)  # pack t to x
        if mask is not None:
            mask = F.pad(mask, (1, 0), value=1)

        rope = self.rotary_embed.forward_from_seq_len(seq_len + 1)

        # flat unet transformer
        skip_connect_type = self.skip_connect_type
        skips = []
        for idx, (maybe_skip_proj, attn_norm, attn, ff_norm, ff) in enumerate(self.layers):
            layer = idx + 1

            # skip connection logic
            is_first_half = layer <= (self.depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half:
                skip = skips.pop()
                if skip_connect_type == "concat":
                    x = torch.cat((x, skip), dim=-1)
                    x = maybe_skip_proj(x)
                elif skip_connect_type == "add":
                    x = x + skip

            # attention and feedforward blocks
            x = attn(attn_norm(x), rope=rope, mask=mask) + x
            x = ff(ff_norm(x)) + x

        assert len(skips) == 0

        x = self.norm_out(x)[:, 1:, :]  # unpack t from x

        return self.proj_out(x)
