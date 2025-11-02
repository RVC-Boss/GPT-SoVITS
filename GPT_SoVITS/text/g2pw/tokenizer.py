from dataclasses import dataclass

import torch
from transformers import BertTokenizerFast


Tensor = torch.Tensor


@dataclass(slots=True)
class G2PWInput:
    input_ids: Tensor
    phoneme_masks: Tensor
    char_ids: Tensor
    position_ids: Tensor


class G2PWTokenizer:
    def __init__(
        self,
        model_source: str,
        labels: list[str],
        char2phonemes: dict[str, list[int]],
        chars: list[str],
        device: torch.device,
    ):
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(model_source, do_lower_case=True)
        self.labels = labels
        self.char2phonemes = char2phonemes
        self.chars = chars
        self.device = device if device.type != "mps" else torch.device("cpu")

    def tokenize(
        self,
        text: str,
        query_ids: list[int],
    ) -> G2PWInput:
        with self.device:
            tokenizer = self.tokenizer

            tokens = list(text)
            processed_tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id = torch.tensor(tokenizer.convert_tokens_to_ids(processed_tokens))

            phoneme_masks: list[Tensor] = []
            char_ids: list[int] = []

            for query_id in query_ids:
                query_char = text[query_id]
                char_id = self.chars.index(query_char)
                mask = torch.zeros(len(self.labels)).bool()
                mask[self.char2phonemes[query_char]] = 1

                phoneme_masks.append(mask)
                char_ids.append(char_id)

            outputs = G2PWInput(
                input_ids=input_id.to(torch.int64).unsqueeze(0).expand(len(query_ids), -1),
                phoneme_masks=torch.stack(phoneme_masks),
                char_ids=torch.tensor(char_ids).to(torch.int64),
                position_ids=torch.tensor(query_ids).to(torch.int64) + 1,  # [CLS] token locate at first place
            )
            return outputs
