import os
import sys
import threading
import time
from contextlib import contextmanager

from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
from text.LangSegmenter import LangSegmenter
from text import chinese
from typing import Dict, List, Tuple
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method
from TTS_infer_pack.prepare_bert_batch_worker import PrepareBertBatchWorker

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


class StageLimiter:
    def __init__(self, slots: int):
        self.slots = max(1, int(slots))
        self.semaphore = threading.BoundedSemaphore(self.slots)
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak_inflight = 0

    @contextmanager
    def enter(self):
        wait_start = time.perf_counter()
        self.semaphore.acquire()
        wait_ms = (time.perf_counter() - wait_start) * 1000.0
        with self.lock:
            self.inflight += 1
            current_inflight = self.inflight
            if current_inflight > self.peak_inflight:
                self.peak_inflight = current_inflight
            peak_inflight = self.peak_inflight
        try:
            yield {
                "wait_ms": wait_ms,
                "inflight": current_inflight,
                "peak_inflight": peak_inflight,
                "slots": self.slots,
            }
        finally:
            with self.lock:
                self.inflight = max(0, self.inflight - 1)
            self.semaphore.release()

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            return {
                "slots": self.slots,
                "inflight": self.inflight,
                "peak_inflight": self.peak_inflight,
            }


class TextPreprocessor:
    def __init__(
        self,
        bert_model: AutoModelForMaskedLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        bert_stage_limiter: StageLimiter | None = None,
        bert_batch_worker: PrepareBertBatchWorker | None = None,
    ):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_stage_limiter = bert_stage_limiter
        self.bert_batch_worker = bert_batch_worker

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
        self, text: str, language: str, version: str = "v1", profile: Dict | None = None
    ) -> Tuple[list, torch.Tensor, str]:
        return self.get_phones_and_bert(text, language, version, profile=profile)

    def _split_text_by_language(self, text: str, language: str) -> Tuple[List[str], List[str]]:
        textlist = []
        langlist = []
        if language == "all_zh":
            for tmp in LangSegmenter.getTexts(text, "zh"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_yue":
            for tmp in LangSegmenter.getTexts(text, "zh"):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ja":
            for tmp in LangSegmenter.getTexts(text, "ja"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ko":
            for tmp in LangSegmenter.getTexts(text, "ko"):
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
                    same_group = (tmp["lang"] == "en" and langlist[-1] == "en") or (
                        tmp["lang"] != "en" and langlist[-1] != "en"
                    )
                    if same_group:
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])
        return textlist, langlist

    def get_phones_and_bert(
        self, text: str, language: str, version: str, final: bool = False, profile: Dict | None = None
    ):
        text = re.sub(r' {2,}', ' ', text)
        textlist, langlist = self._split_text_by_language(text, language)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for segment_text, segment_lang in zip(textlist, langlist):
            phones, word2ph, norm_text = self.clean_text_inf(segment_text, segment_lang, version)
            bert = self.get_bert_inf(phones, word2ph, norm_text, segment_lang, profile=profile)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True, profile=profile)

        return phones, bert, norm_text

    def _accumulate_profile(self, profile: Dict | None, key: str, value: float) -> None:
        if profile is None:
            return
        profile[key] = float(profile.get(key, 0.0)) + float(value)

    def _update_profile_peak(self, profile: Dict | None, key: str, value: float) -> None:
        if profile is None:
            return
        profile[key] = float(max(float(profile.get(key, 0.0)), float(value)))

    def get_bert_feature(self, text: str, word2ph: list, profile: Dict | None = None) -> torch.Tensor:
        if self.bert_batch_worker is not None:
            feature, worker_profile = self.bert_batch_worker.submit(text, word2ph)
            self._accumulate_profile(profile, "bert_wait_ms", worker_profile.get("bert_wait_ms", 0.0))
            self._accumulate_profile(profile, "bert_forward_ms", worker_profile.get("bert_forward_ms", 0.0))
            self._accumulate_profile(profile, "bert_tokenize_ms", worker_profile.get("bert_tokenize_ms", 0.0))
            self._accumulate_profile(profile, "bert_scatter_ms", worker_profile.get("bert_scatter_ms", 0.0))
            self._accumulate_profile(profile, "bert_calls", worker_profile.get("bert_calls", 1.0))
            self._update_profile_peak(
                profile, "bert_stage_inflight_peak", worker_profile.get("bert_stage_inflight_peak", 0.0)
            )
            self._update_profile_peak(profile, "bert_batch_size_peak", worker_profile.get("bert_batch_size", 0.0))
            self._update_profile_peak(profile, "bert_batch_tokens_peak", worker_profile.get("bert_batch_tokens", 0.0))
            if profile is not None:
                profile["bert_stage_slots"] = float(worker_profile.get("bert_stage_slots", 0.0))
            return feature

        limiter_stats = {"wait_ms": 0.0, "inflight": 1, "peak_inflight": 1, "slots": 0}
        if self.bert_stage_limiter is None:
            forward_start = time.perf_counter()
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt")
                for i in inputs:
                    inputs[i] = inputs[i].to(self.device)
                res = self.bert_model(**inputs, output_hidden_states=True)
                res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
        else:
            with self.bert_stage_limiter.enter() as limiter_stats:
                forward_start = time.perf_counter()
                with torch.no_grad():
                    inputs = self.tokenizer(text, return_tensors="pt")
                    for i in inputs:
                        inputs[i] = inputs[i].to(self.device)
                    res = self.bert_model(**inputs, output_hidden_states=True)
                    res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
                forward_ms = (time.perf_counter() - forward_start) * 1000.0
        self._accumulate_profile(profile, "bert_wait_ms", limiter_stats["wait_ms"])
        self._accumulate_profile(profile, "bert_forward_ms", forward_ms)
        self._accumulate_profile(profile, "bert_calls", 1.0)
        self._update_profile_peak(profile, "bert_stage_inflight_peak", limiter_stats["peak_inflight"])
        if profile is not None:
            profile["bert_stage_slots"] = float(limiter_stats["slots"])
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def clean_text_inf(self, text: str, language: str, version: str = "v2"):
        language = language.replace("all_", "")
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str, profile: Dict | None = None):
        language = language.replace("all_", "")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph, profile=profile).to(self.device)
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)

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
