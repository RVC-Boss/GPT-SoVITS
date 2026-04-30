import math

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=dropout)
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


class SinePositionalEmbeddingNested(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
        max_batch_size: int = 20,
        max_seq_len: int = 2500,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = nn.Dropout(p=dropout)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.reverse = False
        self.register_buffer(
            "pe", torch.zeros(max_batch_size, max_seq_len, embedding_dim), persistent=False
        )
        self.pe: torch.Tensor
        self.compute_pe()

    def compute_pe(self):
        if self.reverse:
            position = torch.arange(self.max_seq_len - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pe = self.pe
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, input_pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        pe_values = self.pe[torch.arange(batch_size), input_pos - 1]
        return x * self.x_scale + self.alpha * pe_values.unsqueeze(1)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        input_pos = torch.tensor([i.shape[0] for i in x.unbind()])
        pe_values = torch.nested.nested_tensor(
            [self.pe[i, : input_pos[i], :] for i in range(input_pos.size(0))]
        )
        return x * self.x_scale + self.alpha.item() * pe_values
