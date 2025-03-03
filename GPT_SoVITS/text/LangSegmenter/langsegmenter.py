import logging
import re

# jieba静音
import jieba
jieba.setLogLevel(logging.CRITICAL)

# 更改fast_langdetect大模型位置
from pathlib import Path
import fast_langdetect
fast_langdetect.ft_detect.infer.CACHE_DIRECTORY = Path(__file__).parent.parent.parent / "pretrained_models" / "fast_langdetect"

# 防止win下无法读取模型
import os
from typing import Optional
def load_fasttext_model(
        model_path: Path,
        download_url: Optional[str] = None,
        proxy: Optional[str] = None,
):
    """
    Load a FastText model, downloading it if necessary.
    :param model_path: Path to the FastText model file
    :param download_url: URL to download the model from
    :param proxy: Proxy URL for downloading the model
    :return: FastText model
    :raises DetectError: If model loading fails
    """
    if all([
        fast_langdetect.ft_detect.infer.VERIFY_FASTTEXT_LARGE_MODEL,
        model_path.exists(),
        model_path.name == fast_langdetect.ft_detect.infer.FASTTEXT_LARGE_MODEL_NAME,
    ]):
        if not fast_langdetect.ft_detect.infer.verify_md5(model_path, fast_langdetect.ft_detect.infer.VERIFY_FASTTEXT_LARGE_MODEL):
            fast_langdetect.ft_detect.infer.logger.warning(
                f"fast-langdetect: MD5 hash verification failed for {model_path}, "
                f"please check the integrity of the downloaded file from {fast_langdetect.ft_detect.infer.FASTTEXT_LARGE_MODEL_URL}. "
                "\n    This may seriously reduce the prediction accuracy. "
                "If you want to ignore this, please set `fast_langdetect.ft_detect.infer.VERIFY_FASTTEXT_LARGE_MODEL = None` "
            )
    if not model_path.exists():
        if download_url:
            fast_langdetect.ft_detect.infer.download_model(download_url, model_path, proxy)
        if not model_path.exists():
            raise fast_langdetect.ft_detect.infer.DetectError(f"FastText model file not found at {model_path}")

    try:
        # Load FastText model
        if (re.match(r'^[A-Za-z0-9_/\\:.]*$', str(model_path))):
            model = fast_langdetect.ft_detect.infer.fasttext.load_model(str(model_path))
        else:
            python_dir = os.getcwd()
            if (str(model_path)[:len(python_dir)].upper() == python_dir.upper()):
                model = fast_langdetect.ft_detect.infer.fasttext.load_model(os.path.relpath(model_path, python_dir))
            else:
                import tempfile
                import shutil
                with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                    shutil.copyfile(model_path, tmpfile.name)

                model = fast_langdetect.ft_detect.infer.fasttext.load_model(tmpfile.name)
                os.unlink(tmpfile.name)
        return model

    except Exception as e:
        fast_langdetect.ft_detect.infer.logger.warning(f"fast-langdetect:Failed to load FastText model from {model_path}: {e}")
        raise fast_langdetect.ft_detect.infer.DetectError(f"Failed to load FastText model: {e}")

if os.name == 'nt':
    fast_langdetect.ft_detect.infer.load_fasttext_model = load_fasttext_model


from split_lang import LangSplitter


def full_en(text):
    pattern = r'^[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$'
    return bool(re.match(pattern, text))


def full_cjk(text):
    # 来自wiki
    cjk_ranges = [
        (0x4E00, 0x9FFF),        # CJK Unified Ideographs
        (0x3400, 0x4DB5),        # CJK Extension A
        (0x20000, 0x2A6DD),      # CJK Extension B
        (0x2A700, 0x2B73F),      # CJK Extension C
        (0x2B740, 0x2B81F),      # CJK Extension D
        (0x2B820, 0x2CEAF),      # CJK Extension E
        (0x2CEB0, 0x2EBEF),      # CJK Extension F
        (0x30000, 0x3134A),      # CJK Extension G
        (0x31350, 0x323AF),      # CJK Extension H
        (0x2EBF0, 0x2EE5D),      # CJK Extension H
    ]

    pattern = r'[0-9、-〜。！？.!?… ]+$'

    cjk_text = ""
    for char in text:
        code_point = ord(char)
        in_cjk = any(start <= code_point <= end for start, end in cjk_ranges)
        if in_cjk or re.match(pattern, char):
            cjk_text += char
    return cjk_text


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
                # 未知语言检查是否为CJK
                if dict_item['lang'] == 'x':
                    cjk_text = full_cjk(dict_item['text'])
                    if cjk_text:
                        dict_item = {'lang':'zh','text':cjk_text}
                        lang_list = merge_lang(lang_list,dict_item)
                    continue
                else:
                    lang_list = merge_lang(lang_list,dict_item)
                    continue

            # 存在非日韩文夹日韩文
            for _, temp_item in enumerate(temp_list):
                # 未知语言检查是否为CJK
                if temp_item['lang'] == 'x':
                    cjk_text = full_cjk(dict_item['text'])
                    if cjk_text:
                        dict_item = {'lang':'zh','text':cjk_text}
                        lang_list = merge_lang(lang_list,dict_item)
                else:
                    lang_list = merge_lang(lang_list,temp_item)
        return lang_list
    

if __name__ == "__main__":
    text = "MyGO?,你也喜欢まいご吗？"
    print(LangSegmenter.getTexts(text))

    text = "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。"
    print(LangSegmenter.getTexts(text))

