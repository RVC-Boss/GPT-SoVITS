import os
import re
import sys
from typing import Dict, List, Optional, Tuple

now_dir = os.getcwd()
sys.path.append(now_dir)

from text.LangSegmenter import LangSegmenter
from text import cleaned_text_to_sequence
from text.cleaner import clean_text


PreparedTextSegmentPayload = Dict[str, object]


def split_text_by_language(text: str, language: str) -> Tuple[List[str], List[str]]:
    textlist: List[str] = []
    langlist: List[str] = []
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


def clean_text_segment(text: str, language: str, version: str) -> Tuple[List[int], Optional[List[int]], str]:
    normalized_language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, normalized_language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return list(phones), None if word2ph is None else list(word2ph), str(norm_text)


def preprocess_text_segments_payload(
    text: str,
    language: str,
    version: str,
    final: bool = False,
) -> List[PreparedTextSegmentPayload]:
    text = re.sub(r" {2,}", " ", text)
    textlist, langlist = split_text_by_language(text, language)
    payloads: List[PreparedTextSegmentPayload] = []
    total_phones_len = 0
    for segment_text, segment_lang in zip(textlist, langlist):
        phones, word2ph, norm_text = clean_text_segment(segment_text, segment_lang, version)
        payloads.append(
            {
                "language": segment_lang.replace("all_", ""),
                "phones": phones,
                "word2ph": word2ph,
                "norm_text": norm_text,
            }
        )
        total_phones_len += len(phones)

    if not final and total_phones_len < 6:
        return preprocess_text_segments_payload("." + text, language, version, final=True)

    return payloads
