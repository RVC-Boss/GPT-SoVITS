import torch
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from GPT_SoVITS.Accel.PyTorch import nn

Tensor = torch.Tensor


class G2PW(BertPreTrainedModel):
    def __init__(
        self,
        model_source: BertConfig,
        num_labels: int = 1305,
        num_chars: int = 3582,
        num_pos_tags: int = 11,
    ):
        super().__init__(model_source)

        self.num_labels = num_labels
        self.num_chars = num_chars
        self.num_pos_tags = num_pos_tags
        target_size = self.num_labels

        self.bert = BertModel(self.config)
        self.pos_classifier = nn.Linear(self.config.hidden_size, self.num_pos_tags)
        self.descriptor_bias = nn.Embedding(1, target_size)
        self.char_descriptor = nn.Embedding(self.num_chars, target_size)
        self.second_order_descriptor = nn.Embedding(self.num_chars * self.num_pos_tags, target_size)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.eval()

    def _weighted_softmax(self, logits: Tensor, weights: Tensor, eps: float):
        max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
        weighted_exp_logits = torch.exp(logits - max_logits) * weights
        norm = torch.sum(weighted_exp_logits, dim=-1, keepdim=True)
        probs = weighted_exp_logits / norm
        probs = torch.clamp(probs, min=eps, max=1 - eps)
        return probs

    def _get_char_pos_ids(self, char_ids: Tensor, pos_ids: Tensor):
        return char_ids * self.num_pos_tags + pos_ids

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        phoneme_mask: Tensor,
        char_ids: Tensor,
        position_ids: Tensor,
        eps: float = 1e-6,
    ):
        output = self.bert.forward(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        assert isinstance(output, BaseModelOutputWithPoolingAndCrossAttentions)
        sequence_output = output.last_hidden_state
        assert sequence_output is not None

        batch_size = input_ids.size(0)

        orig_selected_hidden = sequence_output[torch.arange(batch_size), position_ids]

        selected_hidden = orig_selected_hidden

        pred_pos_ids = self.pos_classifier(orig_selected_hidden).argmax(dim=-1)  # teacher mode while training

        bias_tensor = self.descriptor_bias(torch.zeros_like(char_ids))

        char_pos_ids = self._get_char_pos_ids(char_ids, pred_pos_ids)

        affect_hidden = bias_tensor + self.char_descriptor(char_ids) + self.second_order_descriptor(char_pos_ids)

        phoneme_mask = phoneme_mask * torch.sigmoid(affect_hidden)

        logits = self.classifier(selected_hidden)

        probs = self._weighted_softmax(logits, phoneme_mask, eps)

        return probs
