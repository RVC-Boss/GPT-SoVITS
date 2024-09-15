# reference: https://huggingface.co/spaces/Naozumi0512/Bert-VITS2-Cantonese-Yue/blob/main/text/chinese.py

import sys
import re
import cn2an

from pyjyutping import jyutping
from text.symbols import punctuation
from text.zh_normalization.text_normlization import TextNormalizer

normalizer = lambda x: cn2an.transform(x, "an2cn")

INITIALS = [
    "aa",
    "aai",
    "aak",
    "aap",
    "aat",
    "aau",
    "ai",
    "au",
    "ap",
    "at",
    "ak",
    "a",
    "p",
    "b",
    "e",
    "ts",
    "t",
    "dz",
    "d",
    "kw",
    "k",
    "gw",
    "g",
    "f",
    "h",
    "l",
    "m",
    "ng",
    "n",
    "s",
    "y",
    "w",
    "c",
    "z",
    "j",
    "ong",
    "on",
    "ou",
    "oi",
    "ok",
    "o",
    "uk",
    "ung",
]
INITIALS += ["sp", "spl", "spn", "sil"]


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
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def replace_punctuation(text):
    # text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


def text_normalize(text):
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)
    return dest_text


punctuation_set=set(punctuation)
def jyuping_to_initials_finals_tones(jyuping_syllables):
    initials_finals = []
    tones = []
    word2ph = []

    for syllable in jyuping_syllables:
        if syllable in punctuation:
            initials_finals.append(syllable)
            tones.append(0)
            word2ph.append(1)  # Add 1 for punctuation
        elif syllable == "_":
            initials_finals.append(syllable)
            tones.append(0)
            word2ph.append(1)  # Add 1 for underscore
        else:
            try:
                tone = int(syllable[-1])
                syllable_without_tone = syllable[:-1]
            except ValueError:
                tone = 0
                syllable_without_tone = syllable

            for initial in INITIALS:
                if syllable_without_tone.startswith(initial):
                    if syllable_without_tone.startswith("nga"):
                        initials_finals.extend(
                            [
                                syllable_without_tone[:2],
                                syllable_without_tone[2:] or syllable_without_tone[-1],
                            ]
                        )
                        # tones.extend([tone, tone])
                        tones.extend([-1, tone])
                        word2ph.append(2)
                    else:
                        final = syllable_without_tone[len(initial) :] or initial[-1]
                        initials_finals.extend([initial, final])
                        # tones.extend([tone, tone])
                        tones.extend([-1, tone])
                        word2ph.append(2)
                    break
    assert len(initials_finals) == len(tones)

    ###魔改为辅音+带音调的元音
    phones=[]
    for a,b in zip(initials_finals,tones):
        if(b not in [-1,0]):###防止粤语和普通话重合开头加Y，如果是标点，不加。
            todo="%s%s"%(a,b)
        else:todo=a
        if(todo not in punctuation_set):todo="Y%s"%todo
        phones.append(todo)

    # return initials_finals, tones, word2ph
    return phones, word2ph


def get_jyutping(text):
    jp = jyutping.convert(text)
    # print(1111111,jp)
    for symbol in punctuation:
        jp = jp.replace(symbol, " " + symbol + " ")
    jp_array = jp.split()
    return jp_array


def get_bert_feature(text, word2ph):
    from text import chinese_bert

    return chinese_bert.get_bert_feature(text, word2ph)


def g2p(text):
    # word2ph = []
    jyuping = get_jyutping(text)
    # print(jyuping)
    # phones, tones, word2ph = jyuping_to_initials_finals_tones(jyuping)
    phones, word2ph = jyuping_to_initials_finals_tones(jyuping)
    # phones = ["_"] + phones + ["_"]
    # tones = [0] + tones + [0]
    # word2ph = [1] + word2ph + [1]
    return phones, word2ph


if __name__ == "__main__":
    # text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = "佢個鋤頭太短啦。"
    text = text_normalize(text)
    # phones, tones, word2ph = g2p(text)
    phones, word2ph = g2p(text)
    # print(phones, tones, word2ph)
    print(phones, word2ph)