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
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pe = torch.zeros(x.size(1), self.embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * pe
        return self.dropout(output)
