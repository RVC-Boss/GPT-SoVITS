import json
import locale
import os

I18N_JSON_DIR : os.PathLike = os.path.join(os.path.dirname(os.path.relpath(__file__)), 'locale')

def load_language_list(language):
    with open(os.path.join(I18N_JSON_DIR, f"{language}.json"), "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list

def scan_language_list():
    language_list = []
    for name in os.listdir(I18N_JSON_DIR):
        if name.endswith(".json"):language_list.append(name.split('.')[0])
    return language_list

class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[0]  
            # getlocale can't identify the system's language ((None, None))
        if not os.path.exists(os.path.join(I18N_JSON_DIR, f"{language}.json")):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language

if __name__ == "__main__":
    i18n = I18nAuto(language='en_US')
    print(i18n)