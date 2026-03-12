import asyncio
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
from text.LangSegmenter import LangSegmenter
from text import chinese
from typing import Dict, List, Optional, Tuple
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method
from TTS_infer_pack.prepare_bert_batch_worker import PrepareBertBatchWorker
from TTS_infer_pack.text_cpu_preprocess import preprocess_text_segments_payload

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


@dataclass
class PreparedTextSegment:
    language: str
    phones: List[int]
    word2ph: Optional[List[int]]
    norm_text: str


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

    def snapshot(self) -> Dict[str, object]:
        return {
            "device": str(self.device),
            "bert_stage_limiter": (
                None if self.bert_stage_limiter is None else dict(self.bert_stage_limiter.snapshot())
            ),
            "bert_batch_worker": None if self.bert_batch_worker is None else dict(self.bert_batch_worker.snapshot()),
        }

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
            if not re.sub(r"\W+", "", text):
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
        prepared_segments = self.preprocess_text_segments(text, language, version)
        return self.build_phones_and_bert_from_segments(prepared_segments, profile=profile)

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
        prepared_segments = self.preprocess_text_segments(text, language, version, final=final)
        return self.build_phones_and_bert_from_segments(prepared_segments, profile=profile)

    def preprocess_text_segments(
        self,
        text: str,
        language: str,
        version: str,
        final: bool = False,
    ) -> List[PreparedTextSegment]:
        payloads = preprocess_text_segments_payload(text, language, version, final=final)
        return [
            PreparedTextSegment(
                language=str(payload["language"]),
                phones=list(payload["phones"]),
                word2ph=None if payload["word2ph"] is None else list(payload["word2ph"]),
                norm_text=str(payload["norm_text"]),
            )
            for payload in payloads
        ]

    def build_phones_and_bert_from_segments(
        self,
        prepared_segments: List[PreparedTextSegment],
        profile: Dict | None = None,
    ) -> Tuple[list, torch.Tensor, str]:
        phones_list: List[List[int]] = []
        bert_list: List[torch.Tensor] = []
        norm_text_list: List[str] = []
        for segment in prepared_segments:
            bert = self.get_bert_inf(
                segment.phones,
                segment.word2ph,
                segment.norm_text,
                segment.language,
                profile=profile,
            )
            phones_list.append(segment.phones)
            norm_text_list.append(segment.norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)
        return phones, bert, norm_text

    def _accumulate_profile(self, profile: Dict | None, key: str, value: float) -> None:
        if profile is None:
            return
        profile[key] = float(profile.get(key, 0.0)) + float(value)

    def _update_profile_peak(self, profile: Dict | None, key: str, value: float) -> None:
        if profile is None:
            return
        profile[key] = float(max(float(profile.get(key, 0.0)), float(value)))

    def _merge_bert_worker_profile(self, profile: Dict | None, worker_profile: Dict[str, float]) -> None:
        self._accumulate_profile(profile, "bert_wait_ms", worker_profile.get("bert_wait_ms", 0.0))
        self._accumulate_profile(profile, "bert_admission_wait_ms", worker_profile.get("bert_admission_wait_ms", 0.0))
        self._accumulate_profile(profile, "bert_queue_wait_ms", worker_profile.get("bert_queue_wait_ms", 0.0))
        self._accumulate_profile(
            profile,
            "bert_batch_collect_wait_ms",
            worker_profile.get("bert_batch_collect_wait_ms", 0.0),
        )
        self._accumulate_profile(profile, "bert_forward_ms", worker_profile.get("bert_forward_ms", 0.0))
        self._accumulate_profile(profile, "bert_tokenize_ms", worker_profile.get("bert_tokenize_ms", 0.0))
        self._accumulate_profile(profile, "bert_scatter_ms", worker_profile.get("bert_scatter_ms", 0.0))
        self._accumulate_profile(profile, "bert_calls", worker_profile.get("bert_calls", 1.0))
        self._update_profile_peak(profile, "bert_stage_inflight_peak", worker_profile.get("bert_stage_inflight_peak", 0.0))
        self._update_profile_peak(profile, "bert_batch_size_peak", worker_profile.get("bert_batch_size", 0.0))
        self._update_profile_peak(profile, "bert_batch_tokens_peak", worker_profile.get("bert_batch_tokens", 0.0))
        self._update_profile_peak(
            profile,
            "bert_pending_depth_on_enqueue_peak",
            worker_profile.get("bert_pending_depth_on_enqueue", 0.0),
        )
        self._update_profile_peak(
            profile,
            "bert_pending_depth_on_collect_peak",
            worker_profile.get("bert_pending_depth_on_collect", 0.0),
        )
        self._update_profile_peak(profile, "bert_high_pressure_mode_peak", worker_profile.get("bert_high_pressure_mode", 0.0))
        if profile is not None:
            profile["bert_stage_slots"] = float(worker_profile.get("bert_stage_slots", 0.0))
            profile["bert_batch_window_ms"] = float(worker_profile.get("bert_batch_window_ms", 0.0))

    def get_bert_feature(self, text: str, word2ph: list, profile: Dict | None = None) -> torch.Tensor:
        if self.bert_batch_worker is not None:
            feature, worker_profile = self.bert_batch_worker.submit(text, word2ph)
            self._merge_bert_worker_profile(profile, worker_profile)
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

    def get_bert_inf(
        self,
        phones: list,
        word2ph: Optional[list],
        norm_text: str,
        language: str,
        profile: Dict | None = None,
    ):
        language = language.replace("all_", "")
        if language == "zh":
            if word2ph is None:
                raise ValueError("中文文本缺少 word2ph，无法提取 BERT 特征")
            feature = self.get_bert_feature(norm_text, word2ph, profile=profile).to(self.device)
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)

        return feature

    async def build_phones_and_bert_from_segments_async(
        self,
        prepared_segments: List[PreparedTextSegment],
        profile: Dict | None = None,
    ) -> Tuple[list, torch.Tensor, str]:
        segment_jobs = self._build_async_segment_jobs(prepared_segments, profile)
        pending_items: List[Tuple[List[torch.Tensor | None], int, Dict | None, asyncio.Future]] = []
        for segment_index, segment in enumerate(prepared_segments):
            if segment.language.replace("all_", "") != "zh" or self.bert_batch_worker is None:
                continue
            if segment.word2ph is None:
                raise ValueError("中文文本缺少 word2ph，无法提取 BERT 特征")
            pending_items.append(
                (
                    segment_jobs["bert_list"],
                    segment_index,
                    profile,
                    self.bert_batch_worker.submit_async(segment.norm_text, segment.word2ph),
                )
            )

        if pending_items:
            pending_results = await asyncio.gather(*[future for _, _, _, future in pending_items])
            for (bert_list, bert_index, item_profile, _), (feature, worker_profile) in zip(pending_items, pending_results):
                self._merge_bert_worker_profile(item_profile, worker_profile)
                bert_list[bert_index] = feature.to(self.device)

        return self._finalize_async_segment_jobs(segment_jobs)

    def _build_async_segment_jobs(
        self,
        prepared_segments: List[PreparedTextSegment],
        profile: Dict | None,
    ) -> Dict[str, List]:
        phones_list: List[List[int]] = []
        bert_list: List[torch.Tensor | None] = []
        norm_text_list: List[str] = []

        for segment in prepared_segments:
            phones_list.append(segment.phones)
            norm_text_list.append(segment.norm_text)
            segment_language = segment.language.replace("all_", "")
            if segment_language == "zh" and self.bert_batch_worker is not None:
                if segment.word2ph is None:
                    raise ValueError("中文文本缺少 word2ph，无法提取 BERT 特征")
                bert_list.append(None)
                continue
            bert_list.append(
                self.get_bert_inf(
                    segment.phones,
                    segment.word2ph,
                    segment.norm_text,
                    segment.language,
                    profile=profile,
                )
            )
        return {
            "phones_list": phones_list,
            "bert_list": bert_list,
            "norm_text_list": norm_text_list,
        }

    @staticmethod
    def _finalize_async_segment_jobs(segment_jobs: Dict[str, List]) -> Tuple[list, torch.Tensor, str]:
        bert = torch.cat([feature for feature in segment_jobs["bert_list"] if feature is not None], dim=1)
        phones = sum(segment_jobs["phones_list"], [])
        norm_text = "".join(segment_jobs["norm_text_list"])
        return phones, bert, norm_text

    async def build_phones_and_bert_pair_from_segments_async(
        self,
        prompt_segments: List[PreparedTextSegment],
        target_segments: List[PreparedTextSegment],
        prompt_profile: Dict | None = None,
        target_profile: Dict | None = None,
    ) -> Tuple[Tuple[list, torch.Tensor, str], Tuple[list, torch.Tensor, str]]:
        prompt_jobs = self._build_async_segment_jobs(prompt_segments, prompt_profile)
        target_jobs = self._build_async_segment_jobs(target_segments, target_profile)
        pending_items: List[Tuple[List[torch.Tensor | None], int, Dict | None, asyncio.Future]] = []

        for segment_jobs, prepared_segments, profile in (
            (prompt_jobs, prompt_segments, prompt_profile),
            (target_jobs, target_segments, target_profile),
        ):
            for segment_index, segment in enumerate(prepared_segments):
                if segment.language.replace("all_", "") != "zh" or self.bert_batch_worker is None:
                    continue
                if segment.word2ph is None:
                    raise ValueError("中文文本缺少 word2ph，无法提取 BERT 特征")
                pending_items.append(
                    (
                        segment_jobs["bert_list"],
                        segment_index,
                        profile,
                        self.bert_batch_worker.submit_async(segment.norm_text, segment.word2ph),
                    )
                )

        if pending_items:
            pending_results = await asyncio.gather(*[future for _, _, _, future in pending_items])
            for (bert_list, bert_index, profile, _), (feature, worker_profile) in zip(pending_items, pending_results):
                self._merge_bert_worker_profile(profile, worker_profile)
                bert_list[bert_index] = feature.to(self.device)

        return self._finalize_async_segment_jobs(prompt_jobs), self._finalize_async_segment_jobs(target_jobs)

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
