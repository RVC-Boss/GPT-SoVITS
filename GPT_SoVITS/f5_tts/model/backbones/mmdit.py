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

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvPositionEmbedding,
    MMDiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# text embedding


class TextEmbedding(nn.Module):
    def __init__(self, out_dim, text_num_embeds):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, out_dim)  # will use 0 as filler token

        self.precompute_max_pos = 1024
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)

    def forward(self, text: int["b nt"], drop_text=False) -> int["b nt d"]:  # noqa: F722
        text = text + 1
        if drop_text:
            text = torch.zeros_like(text)
        text = self.text_embed(text)

        # sinus pos emb
        batch_start = torch.zeros((text.shape[0],), dtype=torch.long)
        batch_text_len = text.shape[1]
        pos_idx = get_pos_embed_indices(batch_start, batch_text_len, max_pos=self.precompute_max_pos)
        text_pos_embed = self.freqs_cis[pos_idx]

        text = text + text_pos_embed

        return text


# noised input & masked cond audio embedding


class AudioEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = torch.cat((x, cond), dim=-1)
        x = self.linear(x)
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using MM-DiT blocks


class MMDiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        text_num_embeds=256,
        mel_dim=100,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(dim, text_num_embeds)
        self.audio_embed = AudioEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    context_pre_only=i == depth - 1,
                )
                for i in range(depth)
            ]
        )
        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
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
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning (time), c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        c = self.text_embed(text, drop_text=drop_text)
        x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)

        seq_len = x.shape[1]
        text_len = text.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        for block in self.transformer_blocks:
            c, x = block(x, c, t, mask=mask, rope=rope_audio, c_rope=rope_text)

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
