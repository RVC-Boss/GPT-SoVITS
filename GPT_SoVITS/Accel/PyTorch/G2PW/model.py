import math
from pathlib import Path
from typing import MutableSequence

import torch
from transformers import BertConfig

from .. import nn
from ..nn import functional as F

Tensor = torch.Tensor


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
    ):
        super().__init__()

        assert dims % num_heads == 0, "The input feature dimensions should be divisible by the number of heads"

        self.num_heads = num_heads
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)
        self.scale = math.sqrt(1 / dims)

    def __call__(self, x: Tensor):
        B, L, _ = x.shape
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        num_heads = self.num_heads
        queries = queries.view(B, L, num_heads, -1).transpose(1, 2)
        keys = keys.view(B, L, num_heads, -1).transpose(1, 2)
        values = values.view(B, L, num_heads, -1).transpose(1, 2)
        output = F.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(output)


class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: int,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(dims, num_heads)
        self.attention_norm = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.gelu = nn.GELU()

    def __call__(self, x: Tensor):
        attention_out = self.attention(x)
        add_and_norm = self.attention_norm(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ffn_norm(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, dims: int, num_heads: int, mlp_dims: int):
        super().__init__()
        self.layers: MutableSequence[TransformerEncoderLayer] = nn.ModuleList(
            [TransformerEncoderLayer(dims, num_heads, mlp_dims) for _ in range(num_layers)]
        )  # type: ignore

    def __call__(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.requires_grad_(False)

    def __call__(self, input_ids: Tensor):
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings.weight[: input_ids.shape[1]]
        token_types = self.token_type_embeddings.weight[0]

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_dims=config.intermediate_size,
        )

    def __call__(
        self,
        input_ids: Tensor,
    ):
        x = self.embeddings(input_ids)

        y = self.encoder(x)
        return y


class G2PW(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        num_labels: int = 1305,
        num_chars: int = 3582,
        num_pos_tags: int = 11,
    ):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.num_chars = num_chars
        self.num_pos_tags = num_pos_tags
        target_size = self.num_labels

        self.bert = Bert(config)
        self.pos_classifier = nn.Linear(self.config.hidden_size, self.num_pos_tags)
        self.descriptor_bias = nn.Embedding(1, target_size)
        self.char_descriptor = nn.Embedding(self.num_chars, target_size)
        self.second_order_descriptor = nn.Embedding(self.num_chars * self.num_pos_tags, target_size)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

    def _weighted_softmax(self, logits: Tensor, weights: Tensor, eps: float):
        max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
        weighted_exp_logits = torch.exp(logits - max_logits) * weights
        norm = torch.sum(weighted_exp_logits, dim=-1, keepdim=True)
        probs = weighted_exp_logits / norm
        probs = torch.clamp(probs, min=eps, max=1 - eps)

        return probs

    def _get_char_pos_ids(self, char_ids: Tensor, pos_ids: Tensor):
        return char_ids * self.num_pos_tags + pos_ids

    def __call__(
        self,
        input_ids: Tensor,
        phoneme_mask: Tensor,
        char_ids: Tensor,
        position_ids: Tensor,
        eps: float = 1e-6,
    ):
        input_ids = input_ids.to(self.device)
        phoneme_mask = phoneme_mask.to(self.device)
        char_ids = char_ids.to(self.device)
        position_ids = position_ids.to(self.device)

        sequence_output = self.bert(
            input_ids=input_ids,
        )

        orig_selected_hidden = torch.take_along_dim(
            sequence_output,
            position_ids.view(-1, 1, 1),
            dim=1,
        ).squeeze(1)

        pred_pos_ids = self.pos_classifier(orig_selected_hidden).argmax(dim=-1)  # teacher mode while training

        bias_tensor = self.descriptor_bias(torch.zeros_like(char_ids))

        char_pos_ids = self._get_char_pos_ids(char_ids, pred_pos_ids)

        affect_hidden = bias_tensor + self.char_descriptor(char_ids) + self.second_order_descriptor(char_pos_ids)

        phoneme_mask = phoneme_mask * torch.sigmoid(affect_hidden)

        logits = self.classifier(orig_selected_hidden)

        probs = self._weighted_softmax(logits, phoneme_mask, eps)

        pred_idx = probs.argmax(dim=1)

        return pred_idx


def replace_key(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".attention_norm.")
    key = key.replace(".output.LayerNorm.", ".ffn_norm.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    key = key.replace(".LayerNorm.", ".norm.")
    return key


def load_g2pw_torch(bert_model: str, weights_path: str, device: torch.device, dtype: torch.dtype) -> G2PW:
    if not Path(weights_path).exists():
        raise ValueError(f"No model weights found in {weights_path}")

    config = BertConfig.from_pretrained(bert_model)

    # create and update the model
    model = G2PW(config)
    state_dict: dict[str, Tensor] = torch.load(weights_path, map_location="cpu")
    state_dict_torch: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        key = replace_key(key)
        state_dict_torch[key] = value
    model.load_state_dict(state_dict_torch)

    return model.to(device, dtype)
