import logging
import jieba
import re
jieba.setLogLevel(logging.CRITICAL)

# 更改fast_langdetect大模型位置
from pathlib import Path
import fast_langdetect
fast_langdetect.ft_detect.infer.CACHE_DIRECTORY = Path(__file__).parent.parent.parent / "pretrained_models" / "fast_langdetect"
import sys
sys.modules["fast_langdetect"] = fast_langdetect

from split_lang import LangSplitter


def full_en(text):
    pattern = r'^[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$'
    return bool(re.match(pattern, text))


def split_jako(tag_lang,item):
    if tag_lang == "ja":
        pattern = r"([\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]+(?:[0-9、-〜。！？.!?… ]+[\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]*)*)"
    else:
        pattern = r"([\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+(?:[0-9、-〜。！？.!?… ]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]*)*)"

    lang_list: list[dict] = []
    tag = 0
    for match in re.finditer(pattern, item['text']):
        if match.start() > tag:
            lang_list.append({'lang':item['lang'],'text':item['text'][tag:match.start()]})

        tag = match.end()
        lang_list.append({'lang':tag_lang,'text':item['text'][match.start():match.end()]})

    if tag < len(item['text']):
        lang_list.append({'lang':item['lang'],'text':item['text'][tag:len(item['text'])]})

    return lang_list


def merge_lang(lang_list, item):
    if lang_list and item['lang'] == lang_list[-1]['lang']:
        lang_list[-1]['text'] += item['text']
    else:
        lang_list.append(item)
    return lang_list

from typing import List
from split_lang import SubString
def _special_merge_for_zh_ja(
    self,
    substrings: List[SubString],
) -> List[SubString]:
    new_substrings: List[SubString] = []

    if len(substrings) == 1:
        return substrings
    # NOTE: 统计每个语言的字符串长度
    substring_text_len_by_lang = {
        "zh": 0,
        "ja": 0,
        "x": 0,
        "digit": 0,
        "punctuation": 0,
        "newline": 0,
    }
    index = 0
    while index < len(substrings):
        current_block = substrings[index]
        substring_text_len_by_lang[current_block.lang] += current_block.length
        if index == 0:
            if (
                substrings[index + 1].lang in ["zh", "ja"]
                and substrings[index].lang in ["zh", "ja", "x"]
                and substrings[index].length * 10 < substrings[index + 1].length
            ):
                right_block = substrings[index + 1]
                new_substrings.append(
                    SubString(
                        is_digit=False,
                        is_punctuation=False,
                        lang=right_block.lang,
                        text=current_block.text + right_block.text,
                        length=current_block.length + right_block.length,
                        index=current_block.index,
                    )
                )
                index += 1
            else:
                new_substrings.append(substrings[index])
        elif index == len(substrings) - 1:
            left_block = new_substrings[-1]
            if (
                left_block.lang in ["zh", "ja"]
                and current_block.lang in ["zh", "ja", "x"]
                and current_block.length * 10 < left_block.length
            ):
                new_substrings[-1].text += current_block.text
                new_substrings[-1].length += current_block.length

                index += 1
            else:
                new_substrings.append(substrings[index])
        else:
            if (
                new_substrings[-1].lang == substrings[index + 1].lang
                and new_substrings[-1].lang in ["zh", "ja"]
                # and substrings[index].lang in ["zh", "ja", "x"]
                and substrings[index].lang != "en"
                and substrings[index].length * 10
                < new_substrings[-1].length + substrings[index + 1].length
            ):
                left_block = new_substrings[-1]
                right_block = substrings[index + 1]
                current_block = substrings[index]
                new_substrings[-1].text += current_block.text + right_block.text
                new_substrings[-1].length += (
                    current_block.length + right_block.length
                )
                index += 1
            else:
                new_substrings.append(substrings[index])
        index += 1
    # NOTE: 如果 substring_count 中 存在 x，则将 x 设置为最多的 lang
    if substring_text_len_by_lang["x"] > 0:
        max_lang = max(
            substring_text_len_by_lang, key=substring_text_len_by_lang.get
        )
        for index, substr in enumerate(new_substrings):
            if substr.lang == "x":
                new_substrings[index].lang = max_lang
    # NOTE: 如果 ja 数量是 zh 数量的 10 倍以上，则该 zh 设置为 ja
    if substring_text_len_by_lang["ja"] >= substring_text_len_by_lang["zh"] * 10:
        for index, substr in enumerate(new_substrings):
            if substr.lang == "zh":
                new_substrings[index].lang = "ja"
    new_substrings = self._merge_substrings(substrings=new_substrings)
    return new_substrings



class LangSegmenter():
    # 默认过滤器, 基于gsv目前四种语言
    DEFAULT_LANG_MAP = {
        "zh": "zh",
        "yue": "zh",  # 粤语
        "wuu": "zh",  # 吴语
        "zh-cn": "zh",
        "zh-tw": "x", # 繁体设置为x
        "ko": "ko",
        "ja": "ja",
        "en": "en",
    }

    
    def getTexts(text):
        LangSplitter._special_merge_for_zh_ja = _special_merge_for_zh_ja
        lang_splitter = LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)
        substr = lang_splitter.split_by_lang(text=text)

        lang_list: list[dict] = []

        for _, item in enumerate(substr):
            dict_item = {'lang':item.lang,'text':item.text}

            # 处理短英文被识别为其他语言的问题
            if full_en(dict_item['text']):  
                dict_item['lang'] = 'en'
                lang_list = merge_lang(lang_list,dict_item)
                continue

            # 处理非日语夹日文的问题(不包含CJK)
            ja_list: list[dict] = []
            if dict_item['lang'] != 'ja':
                ja_list = split_jako('ja',dict_item)

            if not ja_list:
                ja_list.append(dict_item)

            # 处理非韩语夹韩语的问题(不包含CJK)
            ko_list: list[dict] = []
            temp_list: list[dict] = []
            for _, ko_item in enumerate(ja_list):
                if ko_item["lang"] != 'ko':
                    ko_list = split_jako('ko',ko_item)

                if ko_list:
                    temp_list.extend(ko_list)
                else:
                    temp_list.append(ko_item)

            # 未存在非日韩文夹日韩文
            if len(temp_list) == 1:
                # 跳过未知语言
                if dict_item['lang'] == 'x':
                    continue
                else:
                    lang_list = merge_lang(lang_list,dict_item)
                    continue

            # 存在非日韩文夹日韩文
            for _, temp_item in enumerate(temp_list):
                # 待观察是否会出现带英文或语言为x的中日英韩文
                if temp_item['lang'] == 'x':
                    continue

                lang_list = merge_lang(lang_list,temp_item)

        return lang_list
    

if __name__ == "__main__":
    text = "MyGO?,你也喜欢まいご吗？"
    print(LangSegmenter.getTexts(text))

    text = "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。"
    print(LangSegmenter.getTexts(text))

