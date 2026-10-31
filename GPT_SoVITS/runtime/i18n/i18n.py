"""Inference log localization."""

import json
import locale
from pathlib import Path


I18N_JSON_DIR = Path(__file__).resolve().parent / "locale"


def load_language_list(language):
    with open(I18N_JSON_DIR / f"{language}.json", "r", encoding="utf-8") as file:
        return json.load(file)


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[0]
            # getlocale cannot identify the system language when it returns (None, None).
        if not (I18N_JSON_DIR / f"{language}.json").exists():
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
