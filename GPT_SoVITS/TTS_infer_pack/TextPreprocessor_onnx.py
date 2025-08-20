import os
import sys

from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

import re
from text.LangSegmenter import LangSegmenter
from typing import Dict, List, Tuple
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import PreTrainedTokenizerFast
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method
import onnxruntime as ort
import numpy as np

from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)
punctuation = set(["!", "?", "…", ",", ".", "-"])


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def merge_short_text_in_array(texts: str, threshold: int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


class TextPreprocessorOnnx:
    def __init__(self, onnx_package: str):
        self.bert_model = ort.InferenceSession(os.path.join(onnx_package, "chinese-roberta-wwm-ext-large.onnx"))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(onnx_package, "tokenizer.json"))

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        print(f"############ {i18n('切分文本')} ############")
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(f"############ {i18n('提取文本Bert特征')} ############")
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang, version)
            if phones is None or norm_text == "":
                continue
            res = {
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def pre_seg_text(self, text: str, lang: str, text_split_method: str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in splits and len(get_first(text)) < 4:
            text = "。" + text if lang != "en" else "." + text
        print(i18n("实际输入的目标文本:"))
        print(text)

        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.filter_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        for text in _texts:
            # 解决输入目标文本的空行导致报错的问题
            if len(text.strip()) == 0:
                continue
            if not re.sub("\W+", "", text):
                # 检测一下，如果是纯符号，就跳过。
                continue
            if text[-1] not in splits:
                text += "。" if lang != "en" else "."

            # 解决句子过长导致Bert报错的问题
            if len(text) > 510:
                texts.extend(split_big_text(text))
            else:
                texts.append(text)

        print(i18n("实际输入的目标文本(切句后):"))
        print(texts)
        return texts

    def segment_and_extract_feature_for_text(
        self, text: str, language: str, version: str = "v1"
    ) -> Tuple[list, np.ndarray, str]:
        return self.get_phones_and_bert(text, language, version)

    def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False):
        text = re.sub(r' {2,}', ' ', text)
        textlist = []
        langlist = []
        if language == "all_zh":
            for tmp in LangSegmenter.getTexts(text,"zh"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_yue":
            for tmp in LangSegmenter.getTexts(text,"zh"):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ja":
            for tmp in LangSegmenter.getTexts(text,"ja"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ko":
            for tmp in LangSegmenter.getTexts(text,"ko"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "en":
            langlist.append("en")
            textlist.append(text)
        elif language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if langlist:
                    if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        # print(textlist)
        # print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
            bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = np.concatenate(bert_list, axis=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert, norm_text

    def get_bert_feature(self, text: str, word2ph: list) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="np")
        [res] = self.bert_model.run(None, {
            "input_ids": inputs["input_ids"]
        })
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = np.repeat(res[i:i+1], word2ph[i], axis=0)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = np.concatenate(phone_level_feature, axis=0)
        return phone_level_feature.T

    def clean_text_inf(self, text: str, language: str, version: str = "v2"):
        language = language.replace("all_", "")
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
        language = language.replace("all_", "")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph)
        else:
            feature = np.zeros(
                (1024, len(phones)),
                dtype=np.float32,
            )

        return feature

    def filter_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError(i18n("请输入有效文本"))
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def replace_consecutive_punctuation(self, text):
        punctuations = "".join(re.escape(p) for p in punctuation)
        pattern = f"([{punctuations}])([{punctuations}])+"
        result = re.sub(pattern, r"\1", text)
        return result