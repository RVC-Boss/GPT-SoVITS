from text import japanese, cleaned_text_to_sequence, english,korean,cantonese
import os
if os.environ.get("version","v1")=="v1":
    from text import chinese
    from text.symbols import symbols
else:
    from text import chinese2 as chinese
    from text.symbols2 import symbols

language_module_map = {"zh": chinese, "ja": japanese, "en": english, "ko": korean,"yue":cantonese}
special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]


def clean_text(text, language):
    if(language not in language_module_map):
        language="en"
        text=" "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    language_module = language_module_map[language]
    if hasattr(language_module,"text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text=text
    if language == "zh" or language=="yue":##########
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [','] * (4 - len(phones)) + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None

    for ph in phones:
        assert ph in symbols
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
