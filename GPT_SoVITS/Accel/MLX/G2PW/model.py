import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import torch
from transformers import BertConfig

Array = mx.array
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

    def __call__(self, x: Array):
        B, L, _ = x.shape
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        num_heads = self.num_heads
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
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

    def __call__(self, x: Array) -> Array:
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
        self.layers = [TransformerEncoderLayer(dims, num_heads, mlp_dims) for _ in range(num_layers)]

    def __call__(self, x: Array):
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

    def __call__(self, input_ids: Array) -> Array:
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
        input_ids: Array,
    ) -> Array:
        x = self.embeddings(input_ids)

        y = self.encoder(x)
        return y


class G2PW(nn.Module):
    def __init__(
        self,
        config,
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

    @mx.compile
    @staticmethod
    def _weighted_softmax(logits: Array, weights: Array, eps: float) -> Array:
        max_logits = mx.max(logits, axis=-1, keepdims=True)
        weighted_exp_logits = mx.exp(logits - max_logits) * weights
        norm = mx.sum(weighted_exp_logits, axis=-1, keepdims=True)
        probs = weighted_exp_logits / norm
        probs = mx.minimum(probs, 1 - eps)
        probs = mx.maximum(probs, eps)

        return probs

    def _get_char_pos_ids(self, char_ids: Array, pos_ids: Array):
        return char_ids * self.num_pos_tags + pos_ids

    def __call__(
        self,
        input_ids_torch: Tensor,
        phoneme_mask_torch: Tensor,
        char_ids_torch: Tensor,
        position_ids_torch: Tensor,
        eps: float = 1e-6,
    ):
        input_ids = mx.array(input_ids_torch.cpu())  # type: ignore
        phoneme_mask = mx.array(phoneme_mask_torch.cpu())  # type: ignore
        char_ids = mx.array(char_ids_torch.cpu())  # type: ignore
        position_ids = mx.array(position_ids_torch.cpu())  # type: ignore

        sequence_output = self.bert(input_ids=input_ids)

        orig_selected_hidden = mx.take_along_axis(
            sequence_output,
            position_ids.reshape(-1, 1, 1),
            axis=1,
        ).squeeze(1)

        pred_pos_ids = self.pos_classifier(orig_selected_hidden).argmax(axis=-1)  # teacher mode while training

        bias_tensor = self.descriptor_bias(mx.zeros_like(char_ids))

        char_pos_ids = self._get_char_pos_ids(char_ids, pred_pos_ids)

        affect_hidden = bias_tensor + self.char_descriptor(char_ids) + self.second_order_descriptor(char_pos_ids)

        phoneme_mask = phoneme_mask * mx.sigmoid(affect_hidden)

        logits = self.classifier(orig_selected_hidden)

        probs: Array = self._weighted_softmax(logits, phoneme_mask, eps)

        pred_idx = probs.argmax(axis=1)

        return torch.tensor(pred_idx)


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
    key = key.replace("pooler.dense.", "pooler.")
    return key


def load_g2pw_mlx(bert_model: str, weights_path: str, device: torch.device, dtype: torch.dtype) -> G2PW:
    if not Path(weights_path).exists():
        raise ValueError(f"No model weights found in {weights_path}")

    config = BertConfig.from_pretrained(bert_model)

    # create and update the model
    model = G2PW(config)
    state_dict = torch.load(weights_path, map_location="cpu")
    state_dict_mlx: list[tuple[str, Array]] = []
    for key, value in state_dict.items():
        key = replace_key(key)
        value_mlx = mx.array(value.to(torch.float32))
        state_dict_mlx.append((key, value_mlx))
    model.load_weights(state_dict_mlx)
    match dtype:
        case torch.float16:
            model.set_dtype(mx.float16)
        case torch.bfloat16:
            model.set_dtype(mx.bfloat16)
        case torch.float32:
            model.set_dtype(mx.float16) if device.type != "cpu" else model.set_dtype(mx.float32)
    nn.quantize(model.bert.encoder, group_size=128, bits=4)
    mx.eval(model.parameters())

    return model
