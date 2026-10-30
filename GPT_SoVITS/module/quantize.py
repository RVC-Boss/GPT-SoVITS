# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Inference interface for checkpoint-compatible residual vector quantizers."""

import torch
from torch import nn

from module.core_vq import ResidualVectorQuantization


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dimension: int = 256, n_q: int = 8, bins: int = 1024):
        super().__init__()
        self.n_q = n_q
        self.vq = ResidualVectorQuantization(
            dim=dimension, codebook_size=bins, num_quantizers=n_q,
        )

    def encode(self, x: torch.Tensor, n_q: int = None, st: int = None) -> torch.Tensor:
        return self.vq.encode(x, n_q=n_q or self.n_q, st=st or 0)

    def decode(self, codes: torch.Tensor, st: int = 0) -> torch.Tensor:
        return self.vq.decode(codes, st=st)
