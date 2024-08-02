import re
from jamo import h2j, j2hcj
import ko_pron
from g2pk2 import G2p
from text import symbols

# This is a list of Korean classifiers preceded by pure Korean numerals.
_korean_classifiers = '군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통'

# List of (hangul, hangul divided) pairs:
_hangul_divided = [(re.compile('%s' % x[0]), x[1]) for x in [
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
    ('ㅘ', 'ㅗㅏ'),
    ('ㅙ', 'ㅗㅐ'),
    ('ㅚ', 'ㅗㅣ'),
    ('ㅝ', 'ㅜㅓ'),
    ('ㅞ', 'ㅜㅔ'),
    ('ㅟ', 'ㅜㅣ'),
    ('ㅢ', 'ㅡㅣ'),
    ('ㅑ', 'ㅣㅏ'),
    ('ㅒ', 'ㅣㅐ'),
    ('ㅕ', 'ㅣㅓ'),
    ('ㅖ', 'ㅣㅔ'),
    ('ㅛ', 'ㅣㅗ'),
    ('ㅠ', 'ㅣㅜ')
]]

# List of (Latin alphabet, hangul) pairs:
_latin_to_hangul = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('a', '에이'),
    ('b', '비'),
    ('c', '시'),
    ('d', '디'),
    ('e', '이'),
    ('f', '에프'),
    ('g', '지'),
    ('h', '에이치'),
    ('i', '아이'),
    ('j', '제이'),
    ('k', '케이'),
    ('l', '엘'),
    ('m', '엠'),
    ('n', '엔'),
    ('o', '오'),
    ('p', '피'),
    ('q', '큐'),
    ('r', '아르'),
    ('s', '에스'),
    ('t', '티'),
    ('u', '유'),
    ('v', '브이'),
    ('w', '더블유'),
    ('x', '엑스'),
    ('y', '와이'),
    ('z', '제트')
]]

# List of (ipa, lazy ipa) pairs:
_ipa_to_lazy_ipa = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('t͡ɕ','ʧ'),
    ('d͡ʑ','ʥ'),
    ('ɲ','n^'),
    ('ɕ','ʃ'),
    ('ʷ','w'),
    ('ɭ','l`'),
    ('ʎ','ɾ'),
    ('ɣ','ŋ'),
    ('ɰ','ɯ'),
    ('ʝ','j'),
    ('ʌ','ə'),
    ('ɡ','g'),
    ('\u031a','#'),
    ('\u0348','='),
    ('\u031e',''),
    ('\u0320',''),
    ('\u0339','')
]]


def fix_g2pk2_error(text):
    new_text = ""
    i = 0
    while i < len(text) - 4:
        if (text[i:i+3] == 'ㅇㅡㄹ' or text[i:i+3] == 'ㄹㅡㄹ') and text[i+3] == ' ' and text[i+4] == 'ㄹ':
            new_text += text[i:i+3] + ' ' + 'ㄴ'
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


def hangul_number(num, sino=True):
    '''Reference https://github.com/Kyubyong/g2pK'''
    num = re.sub(',', '', num)

    if num == '0':
        return '영'
    if not sino and num == '20':
        return '스무'

    digits = '123456789'
    names = '일이삼사오육칠팔구'
    digit2name = {d: n for d, n in zip(digits, names)}

    modifiers = '한 두 세 네 다섯 여섯 일곱 여덟 아홉'
    decimals = '열 스물 서른 마흔 쉰 예순 일흔 여든 아흔'
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}

    spelledout = []
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        if sino:
            if i == 0:
                name = digit2name.get(digit, '')
            elif i == 1:
                name = digit2name.get(digit, '') + '십'
                name = name.replace('일십', '십')
        else:
            if i == 0:
                name = digit2mod.get(digit, '')
            elif i == 1:
                name = digit2dec.get(digit, '')
        if digit == '0':
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)):]
                if ''.join(last_three) == '':
                    spelledout.append('')
                    continue
            else:
                spelledout.append('')
                continue
        if i == 2:
            name = digit2name.get(digit, '') + '백'
            name = name.replace('일백', '백')
        elif i == 3:
            name = digit2name.get(digit, '') + '천'
            name = name.replace('일천', '천')
        elif i == 4:
            name = digit2name.get(digit, '') + '만'
            name = name.replace('일만', '만')
        elif i == 5:
            name = digit2name.get(digit, '') + '십'
            name = name.replace('일십', '십')
        elif i == 6:
            name = digit2name.get(digit, '') + '백'
            name = name.replace('일백', '백')
        elif i == 7:
            name = digit2name.get(digit, '') + '천'
            name = name.replace('일천', '천')
        elif i == 8:
            name = digit2name.get(digit, '') + '억'
        elif i == 9:
            name = digit2name.get(digit, '') + '십'
        elif i == 10:
            name = digit2name.get(digit, '') + '백'
        elif i == 11:
            name = digit2name.get(digit, '') + '천'
        elif i == 12:
            name = digit2name.get(digit, '') + '조'
        elif i == 13:
            name = digit2name.get(digit, '') + '십'
        elif i == 14:
            name = digit2name.get(digit, '') + '백'
        elif i == 15:
            name = digit2name.get(digit, '') + '천'
        spelledout.append(name)
    return ''.join(elem for elem in spelledout)


def number_to_hangul(text):
    '''Reference https://github.com/Kyubyong/g2pK'''
    tokens = set(re.findall(r'(\d[\d,]*)([\uac00-\ud71f]+)', text))
    for token in tokens:
        num, classifier = token
        if classifier[:2] in _korean_classifiers or classifier[0] in _korean_classifiers:
            spelledout = hangul_number(num, sino=False)
        else:
            spelledout = hangul_number(num, sino=True)
        text = text.replace(f'{num}{classifier}', f'{spelledout}{classifier}')
    # digit by digit for remaining digits
    digits = '0123456789'
    names = '영일이삼사오육칠팔구'
    for d, n in zip(digits, names):
        text = text.replace(d, n)
    return text


def korean_to_lazy_ipa(text):
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text=re.sub('[\uac00-\ud7af]+',lambda x:ko_pron.romanise(x.group(0),'ipa').split('] ~ [')[0],text)
    for regex, replacement in _ipa_to_lazy_ipa:
        text = re.sub(regex, replacement, text)
    return text

_g2p=G2p()
def korean_to_ipa(text):
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = _g2p(text)
    text = fix_g2pk2_error(text)
    text = korean_to_lazy_ipa(text)
    return text.replace('ʧ','tʃ').replace('ʥ','dʑ')

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
    text = re.sub(r'([\u3131-\u3163])$', r'\1.', text)
    # text = "".join([post_replace_ph(i) for i in text])
    text = [post_replace_ph(i) for i in text]
    return text