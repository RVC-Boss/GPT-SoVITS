# reference: https://github.com/ORI-Muchim/MB-iSTFT-VITS-Korean/blob/main/text/korean.py

import re
from jamo import h2j, j2hcj
from g2pk2 import G2p as BaseG2p
from mecab import MeCab


class KoreanG2p(BaseG2p):
    """Use the same MeCab backend on Windows and Linux."""

    def check_mecab(self):
        # The backend is an explicit runtime dependency, loaded above.
        return None

    def get_mecab(self):
        return MeCab()


from text.symbols2 import symbols


# List of (hangul, hangul divided) pairs:
_hangul_divided = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        # ('ㄳ', 'ㄱㅅ'),   # g2pk2, A Syllable-ending Rule
        # ('ㄵ', 'ㄴㅈ'),
        # ('ㄶ', 'ㄴㅎ'),
        # ('ㄺ', 'ㄹㄱ'),
        # ('ㄻ', 'ㄹㅁ'),
        # ('ㄼ', 'ㄹㅂ'),
        # ('ㄽ', 'ㄹㅅ'),
        # ('ㄾ', 'ㄹㅌ'),
        # ('ㄿ', 'ㄹㅍ'),
        # ('ㅀ', 'ㄹㅎ'),
        # ('ㅄ', 'ㅂㅅ'),
        ("ㅘ", "ㅗㅏ"),
        ("ㅙ", "ㅗㅐ"),
        ("ㅚ", "ㅗㅣ"),
        ("ㅝ", "ㅜㅓ"),
        ("ㅞ", "ㅜㅔ"),
        ("ㅟ", "ㅜㅣ"),
        ("ㅢ", "ㅡㅣ"),
        ("ㅑ", "ㅣㅏ"),
        ("ㅒ", "ㅣㅐ"),
        ("ㅕ", "ㅣㅓ"),
        ("ㅖ", "ㅣㅔ"),
        ("ㅛ", "ㅣㅗ"),
        ("ㅠ", "ㅣㅜ"),
    ]
]

# List of (Latin alphabet, hangul) pairs:
_latin_to_hangul = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("a", "에이"),
        ("b", "비"),
        ("c", "시"),
        ("d", "디"),
        ("e", "이"),
        ("f", "에프"),
        ("g", "지"),
        ("h", "에이치"),
        ("i", "아이"),
        ("j", "제이"),
        ("k", "케이"),
        ("l", "엘"),
        ("m", "엠"),
        ("n", "엔"),
        ("o", "오"),
        ("p", "피"),
        ("q", "큐"),
        ("r", "아르"),
        ("s", "에스"),
        ("t", "티"),
        ("u", "유"),
        ("v", "브이"),
        ("w", "더블유"),
        ("x", "엑스"),
        ("y", "와이"),
        ("z", "제트"),
    ]
]


def fix_g2pk2_error(text):
    new_text = ""
    i = 0
    while i < len(text) - 4:
        if (text[i : i + 3] == "ㅇㅡㄹ" or text[i : i + 3] == "ㄹㅡㄹ") and text[i + 3] == " " and text[i + 4] == "ㄹ":
            new_text += text[i : i + 3] + " " + "ㄴ"
            i += 5
        else:
            new_text += text[i]
            i += 1

    new_text += text[i:]
    return new_text


def latin_to_hangul(text):
    for regex, replacement in _latin_to_hangul:
        text = re.sub(regex, replacement, text)
    return text


def divide_hangul(text):
    text = j2hcj(h2j(text))
    for regex, replacement in _hangul_divided:
        text = re.sub(regex, replacement, text)
    return text


_g2p = KoreanG2p()


def post_replace_ph(ph):
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
        " ": "空",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "停"
    return ph


def g2p(text):
    text = latin_to_hangul(text)
    text = _g2p(text)
    text = divide_hangul(text)
    text = fix_g2pk2_error(text)
    text = re.sub(r"([\u3131-\u3163])$", r"\1.", text)
    # text = "".join([post_replace_ph(i) for i in text])
    text = [post_replace_ph(i) for i in text]
    return text
