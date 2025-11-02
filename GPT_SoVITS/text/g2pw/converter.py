# This code is modified from https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw
# This code is modified from https://github.com/GitYCC/g2pW

import importlib.util
import json
import os
import platform
from typing import Callable

import torch
from opencc import OpenCC
from pypinyin import Style, pinyin

from config import get_dtype, infer_device

from ..zh_normalization.char_convert import traditional_to_simplified
from .tokenizer import G2PWInput, G2PWTokenizer

device = infer_device
dtype = get_dtype(device.index)


def load_g2pw(bert_model_source: str, device: torch.device, dtype: torch.dtype):
    match device.type:
        case "mps" if importlib.util.find_spec("mlx") is not None and platform.system() == "Darwin":
            from GPT_SoVITS.Accel.MLX import load_g2pw_mlx

            return load_g2pw_mlx(bert_model_source, bert_model_source + "/pytorch_model.bin", device, dtype)
        case "cpu" | "cuda" | "xpu" | "mtia" | "mps":
            from GPT_SoVITS.Accel.PyTorch import load_g2pw_torch

            return load_g2pw_torch(bert_model_source, bert_model_source + "/pytorch_model.bin", device, dtype)
        case _:
            raise ValueError(f"Unsupported device type: {device.type}")


class G2PWConverter:
    def __init__(
        self,
        model_source: str = "GPT_SoVITS/pretrained_models/g2pw-chinese",
        style: str = "bopomofo",
    ):
        self.model_source = model_source
        assert os.path.exists(model_source), f"model_source: {model_source} not found."

        polyphonic_chars_path = os.path.join(model_source, "POLYPHONIC_CHARS.txt")
        monophonic_chars_path = os.path.join(model_source, "MONOPHONIC_CHARS.txt")
        self.polyphonic_chars = [
            line.split("\t") for line in open(polyphonic_chars_path, encoding="utf-8").read().strip().split("\n")
        ]
        self.non_polyphonic = {
            "一",
            "不",
            "和",
            "咋",
            "嗲",
            "剖",
            "差",
            "攢",
            "倒",
            "難",
            "奔",
            "勁",
            "拗",
            "肖",
            "瘙",
            "誒",
            "泊",
            "听",
            "噢",
        }
        self.non_monophonic = {"似", "攢"}
        self.monophonic_chars = [
            line.split("\t") for line in open(monophonic_chars_path, encoding="utf-8").read().strip().split("\n")
        ]
        self.labels, self.char2phonemes = self.get_phoneme_labels(polyphonic_chars=self.polyphonic_chars)

        self.chars = sorted(list(self.char2phonemes.keys()))

        self.polyphonic_chars_new = set(self.chars)

        for char in self.non_polyphonic:
            if char in self.polyphonic_chars_new:
                self.polyphonic_chars_new.remove(char)

        self.monophonic_chars_dict = {char: phoneme for char, phoneme in self.monophonic_chars}
        for char in self.non_monophonic:
            if char in self.monophonic_chars_dict:
                self.monophonic_chars_dict.pop(char)

        self.pos_tags = ["UNK", "A", "C", "D", "I", "N", "P", "T", "V", "DE", "SHI"]

        with open(os.path.join(model_source, "bopomofo_to_pinyin_wo_tune_dict.json"), "r", encoding="utf-8") as fr:
            self.bopomofo_convert_dict: dict[str, str] = json.load(fr)

        self.style_convert_func: Callable[[str], str] = {
            "bopomofo": lambda x: x,
            "pinyin": self._convert_bopomofo_to_pinyin,
        }[style]

        with open(os.path.join(model_source, "char_bopomofo_dict.json"), "r", encoding="utf-8") as fr:
            self.char_bopomofo_dict = json.load(fr)

        self.cc = OpenCC("s2tw")

        self.init_model = False

    def _convert_bopomofo_to_pinyin(self, bopomofo: str) -> str:
        tone = bopomofo[-1]
        assert tone in "12345"
        component = self.bopomofo_convert_dict.get(bopomofo[:-1])
        if component:
            return component + tone
        else:
            raise ValueError(f'Cannot convert "{bopomofo}" to pinyin')

    def predict(self, input: G2PWInput):
        labels = self.labels

        pred_idx = self.g2pw_model(
            input.input_ids,
            input.phoneme_masks,
            input.char_ids,
            input.position_ids,
        )

        pred_labels: list[str] = [labels[i] for i in pred_idx.tolist()]

        return pred_labels

    def __call__(self, sentence: str) -> list[list[str]]:
        if not self.init_model:
            self.g2pw_model = load_g2pw(self.model_source, device, dtype)
            self.tokenizer = G2PWTokenizer(
                model_source=self.model_source,
                labels=self.labels,
                char2phonemes=self.char2phonemes,
                chars=self.chars,
                device=device,
            )
            self.init_model = True

        translated_sent = self.cc.convert(sentence)
        assert len(translated_sent) == len(sentence)
        sentence = translated_sent

        query_ids, partial_result = self._prepare_data(translated_sent)

        if len(query_ids) == 0:
            return [partial_result]

        model_input = self.tokenizer.tokenize(translated_sent, query_ids)

        preds = self.predict(model_input)

        for qid, pred in zip(query_ids, preds):
            partial_result[qid] = self.style_convert_func(pred)

        return [partial_result]

    def _prepare_data(self, sentence: str) -> tuple[list[int], list[str]]:
        texts: list[str] = []
        query_ids: list[int] = []

        sentence_s = traditional_to_simplified(sentence)
        pypinyin_result = pinyin(sentence_s, neutral_tone_with_five=True, style=Style.TONE3)

        partial_result: list[str] = [""] * len(sentence)

        for i, char in enumerate(sentence):
            if char in self.polyphonic_chars_new:
                texts.append(sentence)
                query_ids.append(i)
            elif char in self.monophonic_chars_dict:
                partial_result[i] = self.style_convert_func(self.monophonic_chars_dict[char])
            elif char in self.char_bopomofo_dict:
                partial_result[i] = pypinyin_result[i][0]
            else:
                partial_result[i] = pypinyin_result[i][0]

        return query_ids, partial_result

    def get_phoneme_labels(self, polyphonic_chars: list[list[str]]):
        labels = sorted(list(set([phoneme for _, phoneme in polyphonic_chars])))
        char2phonemes: dict[str, list[int]] = {}
        for char, phoneme in polyphonic_chars:
            if char not in char2phonemes:
                char2phonemes[char] = []
            char2phonemes[char].append(labels.index(phoneme))
        return labels, char2phonemes
