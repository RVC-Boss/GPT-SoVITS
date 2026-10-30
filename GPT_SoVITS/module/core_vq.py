# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Inference-only nearest-neighbor and residual vector quantization."""

import torch
from torch import nn
from torch.nn import functional as F


class EuclideanCodebook(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self.register_buffer("embed", torch.zeros(codebook_size, dim))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # EMA statistics are retained in old checkpoints but never used to
        # select or decode tokens. The learned embed buffer remains mandatory.
        for name in ("inited", "cluster_size", "embed_avg"):
            state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def encode(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        embed = self.embed.t()
        distance = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
        return distance.max(dim=-1).indices.view(*shape[:-1])

    def decode(self, indices):
        return F.embedding(indices, self.embed)


class VectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int, codebook_dim: int = None):
        super().__init__()
        codebook_dim = dim if codebook_dim is None else codebook_dim
        self.project_in = nn.Linear(dim, codebook_dim) if codebook_dim != dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if codebook_dim != dim else nn.Identity()
        self._codebook = EuclideanCodebook(codebook_dim, codebook_size)

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        return self._codebook.encode(self.project_in(x.transpose(1, 2)))

    def decode(self, indices):
        return self.project_out(self._codebook.decode(indices)).transpose(1, 2)


class ResidualVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def encode(self, x: torch.Tensor, n_q: int = None, st: int = None) -> torch.Tensor:
        residual = x
        indices = []
        for layer in self.layers[st or 0:n_q or len(self.layers)]:
            codes = layer.encode(residual)
            residual = residual - layer.decode(codes)
            indices.append(codes)
        return torch.stack(indices)

    def decode(self, indices: torch.Tensor, st: int = 0) -> torch.Tensor:
        quantized = torch.tensor(0.0, device=indices.device)
        for index, codes in enumerate(indices):
            quantized = quantized + self.layers[st + index].decode(codes)
        return quantized
