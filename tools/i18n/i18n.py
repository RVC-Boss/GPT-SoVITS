import json
import locale
import os


def load_language_list(language, locale_path="./i18n/locale"):
    with open(os.path.join(locale_path, f"{language}.json"), "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language=None, locale_path="./i18n/locale"):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        if not os.path.exists(os.path.join(locale_path, f"{language}.json")):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language, locale_path)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
