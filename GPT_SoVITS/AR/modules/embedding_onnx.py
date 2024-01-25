# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/embedding.py
import math

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor):
        x = self.word_embeddings(x)
        x = self.dropout(x)
        return x


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.reverse = False
        self.div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * -(math.log(10000.0) / self.embedding_dim))

    def extend_pe(self, x):
        position = torch.cumsum(torch.ones_like(x[:,:,0]), dim=1).transpose(0, 1)
        scpe = (position * self.div_term).unsqueeze(0)
        pe = torch.cat([torch.sin(scpe), torch.cos(scpe)]).permute(1, 2, 0)
        pe = pe.contiguous().view(1, -1, self.embedding_dim)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * pe
        return self.dropout(output)
