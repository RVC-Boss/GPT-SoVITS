import json
import locale
import os

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


def load_language_list(language):
    # 使用从项目根目录开始的绝对路径
    language_file_path = os.path.join(PROJECT_ROOT, f"i18n/locale/{language}.json")
    with open(language_file_path, "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        language_file_path = os.path.join(PROJECT_ROOT, f"i18n/locale/{language}.json")
        if not os.path.exists(language_file_path):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
