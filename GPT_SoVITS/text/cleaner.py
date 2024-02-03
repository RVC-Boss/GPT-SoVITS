from text import chinese, japanese, cleaned_text_to_sequence, symbols, english
import ttsfrd
import re,os,zipfile,requests

ENG_LANG_MAPPING = {
    "PinYin": "zh-cn",
    "English": "en-us",
    "British": "en-gb",
    "ZhHK": "hk_cantonese",
    "Sichuan": "sichuan",
    "Japanese": "japanese",
    "WuuShangHai": "shanghai",
    "Indonesian": "indonesian",
    "Malay": "malay",
    "Filipino": "filipino",
    "Vietnamese": "vietnamese",
    "Korean": "korean",
    "Russian": "russian",
}

chinese_dict = {
    "xx":"x",
    "uei":"ui",
    "ii":"i0",
    "ih":"ir",
    "uen":"un",
    "iou":"iu",
    "angr":"ang",
    "anr":"an",
    "aor":"ao",
    "ar":"a",
    "eir":"ei",
    "engr":"eng",
    "enr":"en",
    "ianr":"ian",
    "iaor":"iao",
    "ingr":"ing",
    "or":"o",
    "ur":"u",
    "ihr":"ih",
    "ongr":"ong",
    "our":"ou",
    "uangr":"uang",
    "uanr":"uan",
    "ueir":"uei",
    "uenr":"uen",
    "uor":"uo",
    "iir":"ii",
    "air":"ai",
    "ier":"ie",
    "uair":"uai",
    "uar":"ua",
    "iar":"ia",
    "inr":"in",
    "iour":"iou",
    "vanr":"van",
    "ver":"ve",
    "vnr":"vn",
    "iangr":"iang",
    "vr":"v",
    "iongr":"iong",
}
english_dict = {
    "DH1":"DH",
    "NG0":"NG",
    "SH0":"SH",
    "NG1":"NG",
    "CH0":"CH",
    "HH0":"HH",
    "ZH0":"ZH",
    "HH1":"HH",
    "SH1":"SH",
    "ZH1":"ZH",
    "DH0":"DH",
    "TH1":"TH",
    "CH1":"CH",
    "JH1":"JH",
    "JH0":"JH",
    "NG2":"NG",
    "TH0":"TH",
}
japanese_dict = {
    "nn":"N",
    "ux":"U",
    "ix":"I",
}

resource_dir = "GPT_SoVITS/text/resource"
resources_zip_file = "GPT_SoVITS/text/resource.zip"
if not os.path.exists(resource_dir):
    if not os.path.exists(resources_zip_file):
        print("Downloading ttsfrd resources...")
        modelscope_url = "https://www.modelscope.cn/api/v1/models/speech_tts/speech_kantts_ttsfrd/repo?Revision=v0.0.1&FilePath=resource.zip"
        with requests.get(modelscope_url, stream=True) as r:
            r.raise_for_status()
            with open(resources_zip_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    
    print("Extracting ttsfrd resources...")
    with zipfile.ZipFile(resources_zip_file, "r") as zip_ref:
        zip_ref.extractall("GPT_SoVITS/text")

fe = ttsfrd.TtsFrontendEngine()
assert fe.initialize(resource_dir),"Check ttsfrd resource"

def clean_text(text, language):
    if(language not in language_module_map):
        language="en"
        text=" "
    if language == "zh":
        phones = []
        word2ph = []
        count = 0
        fe.set_lang_type(ENG_LANG_MAPPING["PinYin"])
        res = fe.gen_tacotron_symbols(text)
        matches = re.findall(r'\{(.*?)\}', res)
        for match in matches:
            elements = match.split("$")
            if elements[2] == "s_none":
                if elements[0] == "#4":
                    phone = "."
                    phones += [phone]
                    word2ph.append(1)
                    continue
                if elements[0] == "#3":
                    phone = ","
                    phones += [phone]
                    word2ph.append(1)
                continue

            # Chinese
            if elements[0] == "ga":
                phone = "AA"
                phones += [phone]
                count += 1
                continue
            if elements[0] == "ge":
                phone = "EE"
                phones += [phone]
                count += 1
                continue
            if elements[0] == "go":
                phone = "OO"
                phones += [phone]
                count += 1
                continue
            if "_c" in elements[0]:
                if elements[2] in ("s_begin","s_middle","s_both","s_end"):
                    phone = elements[0].replace("_c", "")
                    phone = chinese_dict.get(phone, phone)
                    phone = chinese_dict.get(phone, phone)
                    count += 1
                    if elements[2] == "s_end":
                        phone += elements[1].replace("tone", "")
                        word2ph.append(count)
                        count = 0
                    phones += [phone]
                    continue

            # English
            else:
                if elements[2] in ("s_begin","s_middle","s_both","s_end"):
                    phone = elements[0].upper()
                    if len(elements[0]) > 1 :
                        phone += elements[1].replace("tone", "")
                    phone = english_dict.get(phone, phone)
                phones += [phone]
                continue
    elif language == "en":
        phones = []
        word2ph = None
        fe.set_lang_type(ENG_LANG_MAPPING["English"])
        res = fe.gen_tacotron_symbols(text)
        matches = re.findall(r'\{(.*?)\}', res)
        for match in matches:
            elements = match.split("$")
            if elements[2] == "s_none":
                if elements[0] == "#4":
                    phone = "."
                    phones += [phone]
                continue

            if elements[2] in ("s_begin","s_middle","s_both","s_end"):
                phone = elements[0].upper()
                if len(elements[0]) > 1 :
                    phone += elements[1].replace("tone", "")
                phone = english_dict.get(phone, phone)
            phones += [phone]
            continue
    elif language == "ja":
        phones = []
        word2ph = None
        fe.set_lang_type(ENG_LANG_MAPPING["Japanese"])
        res = fe.gen_tacotron_symbols(text)
        matches = re.findall(r'\{(.*?)\}', res)
        for match in matches:
            elements = match.split("$")
            if elements[2] == "s_none":
                if elements[0] == "#4":
                    phone = "."
                    phones += [phone]
                    continue
                if elements[0] == "#3":
                    phone = ","
                    phones += [phone]
                continue
            
            if elements[2] in ("s_begin","s_middle","s_both","s_end"):
                phone = elements[0]
                phone = japanese_dict.get(phone, phone)
            phones += [phone]
            continue
    # print("new:",phones)
    # p,w,n = clean_text_old(text, language)
    # print("old:",p)
    return phones, word2ph, text

language_module_map = {"zh": chinese, "ja": japanese, "en": english}
special = [
    ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]


def clean_text_old(text, language):
    if(language not in language_module_map):
        language="en"
        text=" "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
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
