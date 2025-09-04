from text import cleaned_text_to_sequence
import os

# Only V2+ symbols are supported (V4 and V2Pro use V2+ language set)
from text import symbols2 as symbols_v2


def clean_text(text, language, version=None):
    # Only V2+ versions are supported
    symbols = symbols_v2.symbols
    # Only Korean and English languages supported
    language_module_map = {"en": "english", "ko": "korean"}

    if language not in language_module_map:
        language = "en"
        text = " "
    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text
    if language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [","] + phones
        word2ph = None
    else:  # korean
        phones = language_module.g2p(norm_text)
        word2ph = None
    phones = ["UNK" if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


def text_to_sequence(text, language, version=None):
    version = os.environ.get("version", version)
    if version is None:
        version = "v2"
    phones = clean_text(text, language)
    return cleaned_text_to_sequence(phones, version)


if __name__ == "__main__":
    print(clean_text("Hello world", "en"))
