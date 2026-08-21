import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi
from text.zh_normalization.text_normlization import TextNormalizer

normalizer = lambda x: cn2an.transform(x, "an2cn")

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba_fast
import logging

jieba_fast.setLogLevel(logging.CRITICAL)
import jieba_fast.posseg as psg

# English letter → Chinese pinyin text (how native speakers naturally read abbreviations)
EN_CHAR_TO_CN = {
    'A': '诶', 'B': '必', 'C': '西', 'D': '迪', 'E': '伊',
    'F': '艾弗', 'G': '基', 'H': '艾尺', 'I': '艾',
    'J': '杰', 'K': '开', 'L': '勒', 'M': '艾姆', 'N': '恩',
    'O': '欧', 'P': '批', 'Q': '丘', 'R': '艾尔', 'S': '艾斯',
    'T': '替', 'U': '优', 'V': '维', 'W': '达不溜', 'X': '克斯',
    'Y': '歪', 'Z': '贼',
}


def _replace_en_abbrevs(text):
    """Replace runs of 2+ uppercase English letters with Chinese phonetic equivalents."""
    result = []
    i = 0
    while i < len(text):
        if text[i].isupper() and text[i].isalpha():
            start = i
            while i < len(text) and text[i].isupper() and text[i].isalpha():
                i += 1
            abbr = text[start:i]
            if len(abbr) >= 2:
                result.append(''.join(EN_CHAR_TO_CN.get(ch, ch) for ch in abbr))
            else:
                result.append(abbr)
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)
# is_g2pw = False#True if is_g2pw_str.lower() == 'true' else False
is_g2pw = True  # True if is_g2pw_str.lower() == 'true' else False
if is_g2pw:
    # print("当前使用g2pw进行拼音推理")
    from text.g2pw import G2PWPinyin, correct_pronunciation

    parent_directory = os.path.dirname(current_file_path)
    g2pw = G2PWPinyin(
        model_dir="GPT_SoVITS/text/G2PWModel",
        model_source=os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        v_to_u=False,
        neutral_tone_with_five=True,
    )

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text


def g2p(text):
    text = _replace_en_abbrevs(text)          # English abbrevs → Chinese chars
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, word2ph = _g2p(sentences)
    return phones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []

    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


must_erhua = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}
not_erhua = {
    "虐儿",
    "为儿",
    "护儿",
    "瞒儿",
    "救儿",
    "替儿",
    "有儿",
    "一儿",
    "我儿",
    "俺儿",
    "妻儿",
    "拐儿",
    "聋儿",
    "乞儿",
    "患儿",
    "幼儿",
    "孤儿",
    "婴儿",
    "婴幼儿",
    "连体儿",
    "脑瘫儿",
    "流浪儿",
    "体弱儿",
    "混血儿",
    "蜜雪儿",
    "舫儿",
    "祖儿",
    "美儿",
    "应采儿",
    "可儿",
    "侄儿",
    "孙儿",
    "侄孙儿",
    "女儿",
    "男儿",
    "红孩儿",
    "花儿",
    "虫儿",
    "马儿",
    "鸟儿",
    "猪儿",
    "猫儿",
    "狗儿",
    "少儿",
}


def _merge_erhua(initials: list[str], finals: list[str], word: str, pos: str) -> list[list[str]]:
    """
    Do erhub.
    """
    # fix er1
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn == "er1":
            finals[i] = "er2"

    # 发音
    if word not in must_erhua and (word in not_erhua or pos in {"a", "j", "nr"}):
        return initials, finals

    # "……" 等情况直接返回
    if len(finals) != len(word):
        return initials, finals

    assert len(finals) == len(word)

    # 与前一个字发同音
    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if (
            i == len(finals) - 1
            and word[i] == "儿"
            and phn in {"er2", "er5"}
            and word[-2:] not in not_erhua
            and new_finals
        ):
            phn = "er" + new_finals[-1][-1]

        new_initials.append(initials[i])
        new_finals.append(phn)

    return new_initials, new_finals


def _g2p(segments):
    phones_list = []
    word2ph = []
    g2pw_batch_results = []
    g2pw_batch_cursor = 0
    processed_segments = [re.sub("[a-zA-Z]+", "", seg) for seg in segments]
    if is_g2pw:
        batch_inputs = [seg for seg in processed_segments if seg]
        g2pw_batch_results = g2pw._g2pw(batch_inputs) if batch_inputs else []

    for seg in segments:  # 用原始文本（含英文字母）做 jieba 分词
        pinyins = []
        seg_cut = psg.lcut(seg)
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        initials = []
        finals = []

        if not is_g2pw:
            for word, pos in seg_cut:
                if pos == "eng":
                    # 英文字母直接转 ARPAbet 音素
                    for char in word:
                        if char.isalpha():
                            eng_phones = _g2p_english_char(char.upper())
                            phones_list += eng_phones
                            word2ph.append(len(eng_phones))
                    continue
                sub_initials, sub_finals = _get_initials_finals(word)
                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
                # 儿化
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
                initials.append(sub_initials)
                finals.append(sub_finals)
                # assert len(sub_initials) == len(sub_finals) == len(word)
            initials = sum(initials, [])
            finals = sum(finals, [])
            print("pypinyin结果", initials, finals)
        else:
            # g2pw采用整句推理（批量推理，逐句取结果）
            seg_stripped = processed_segments[g2pw_batch_cursor] if seg else ""
            if seg_stripped:
                pinyins = g2pw_batch_results[g2pw_batch_cursor]
            else:
                pinyins = []
            if seg:
                g2pw_batch_cursor += 1

            pre_word_length = 0
            for word, pos in seg_cut:
                sub_initials = []
                sub_finals = []

                if pos == "eng":
                    # 英文字母缩写 → 逐字母读 ARPAbet 音素
                    for char in word:
                        if char.isalpha():
                            eng_phones = _g2p_english_char(char.upper())
                            phones_list += eng_phones
                            word2ph.append(len(eng_phones))
                    # 英文字母不在 g2pw 的 pinyin 数组里，不推进 pre_word_length
                    continue

                now_word_length = pre_word_length + len(word)
                word_pinyins = pinyins[pre_word_length:now_word_length]

                # 多音字消歧
                word_pinyins = correct_pronunciation(word, word_pinyins)

                for pinyin in word_pinyins:
                    if pinyin[0].isalpha():
                        sub_initials.append(to_initials(pinyin))
                        sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                    else:
                        sub_initials.append(pinyin)
                        sub_finals.append(pinyin)

                pre_word_length = now_word_length
                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
                # 儿化
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
                initials.append(sub_initials)
                finals.append(sub_finals)

            initials = sum(initials, [])
            finals = sum(finals, [])
            # print("g2pw结果",initials,finals)

        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                new_v = new_v + tone
                phone = [new_c, new_v]
                word2ph.append(len(phone))

            phones_list += phone
    return phones_list, word2ph


def _g2p_english_char(char):
    """英文字母 → ARPAbet 音素（字母读音，单字母缩写字面读法）

    示例: N → ['EH1', 'N'], O → ['OW1'], A → ['EY1']
    这些音素在 symbols2.py 符号表里都有定义。
    """
    mapping = {
        'A': ['EY1'], 'B': ['B', 'IY1'], 'C': ['S', 'IY1'],
        'D': ['D', 'IY1'], 'E': ['IY1'], 'F': ['EH1', 'F'],
        'G': ['JH', 'IY1'], 'H': ['EY1', 'CH'], 'I': ['AY1'],
        'J': ['JH', 'EY1'], 'K': ['K', 'EY1'], 'L': ['EH1', 'L'],
        'M': ['EH1', 'M'], 'N': ['EH1', 'N'], 'O': ['OW1'],
        'P': ['P', 'IY1'], 'Q': ['K', 'Y', 'UW1'], 'R': ['AA1', 'R'],
        'S': ['EH1', 'S'], 'T': ['T', 'IY1'], 'U': ['Y', 'UW1'],
        'V': ['V', 'IY1'], 'W': ['D', 'AH1', 'B', 'AH0', 'L', 'Y', 'UW0'],
        'X': ['EH1', 'K', 'S'], 'Y': ['W', 'AY1'],
        'Z': ['Z', 'IY1'],
    }
    return mapping.get(char, ['UNK'])


def replace_punctuation_with_en(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(r"[^\u4e00-\u9fa5A-Za-z" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def text_normalize(text):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        # 使用保留英文字母的版本，让英文字母缩写在中文管道中存活
        dest_text += replace_punctuation_with_en(sentence)

    # 避免重复标点引起的参考泄露
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


if __name__ == "__main__":
    text = "啊——但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏"
    text = "呣呣呣～就是…大人的鼹鼠党吧？"
    text = "你好"
    text = text_normalize(text)
    print(g2p(text))


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试..."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
