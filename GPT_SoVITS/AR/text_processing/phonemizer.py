# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/text_processing/phonemizer.py
# reference: https://github.com/lifeiteng/vall-e
import itertools
import re
from typing import Dict
from typing import List

import regex
from gruut import sentences
from gruut.const import Sentence
from gruut.const import Word
from AR.text_processing.symbols import SYMBOL_TO_ID


class GruutPhonemizer:
    def __init__(self, language: str):
        self._phonemizer = sentences
        self.lang = language
        self.symbol_to_id = SYMBOL_TO_ID
        self._special_cases_dict: Dict[str] = {
            r"\.\.\.": "... ",
            ";": "; ",
            ":": ": ",
            ",": ", ",
            r"\.": ". ",
            "!": "! ",
            r"\?": "? ",
            "—": "—",
            "…": "… ",
            "«": "«",
            "»": "»",
        }
        self._punctuation_regexp: str = rf"([{''.join(self._special_cases_dict.keys())}])"

    def _normalize_punctuation(self, text: str) -> str:
        text = regex.sub(rf"\pZ+{self._punctuation_regexp}", r"\1", text)
        text = regex.sub(rf"{self._punctuation_regexp}(\pL)", r"\1 \2", text)
        text = regex.sub(r"\pZ+", r" ", text)
        return text.strip()

    def _convert_punctuation(self, word: Word) -> str:
        if not word.phonemes:
            return ""
        if word.phonemes[0] in ["‖", "|"]:
            return word.text.strip()

        phonemes = "".join(word.phonemes)
        # remove modifier characters ˈˌː with regex
        phonemes = re.sub(r"[ˈˌː͡]", "", phonemes)
        return phonemes.strip()

    def phonemize(self, text: str, espeak: bool = False) -> str:
        text_to_phonemize: str = self._normalize_punctuation(text)
        sents: List[Sentence] = [sent for sent in self._phonemizer(text_to_phonemize, lang="en-us", espeak=espeak)]
        words: List[str] = [self._convert_punctuation(word) for word in itertools.chain(*sents)]
        return " ".join(words)

    def transform(self, phonemes):
        # convert phonemes to ids
        # dictionary is in symbols.py
        return [self.symbol_to_id[p] for p in phonemes if p in self.symbol_to_id.keys()]


if __name__ == "__main__":
    phonemizer = GruutPhonemizer("en-us")
    # text -> IPA
    phonemes = phonemizer.phonemize("Hello, wor-ld ?")
    print("phonemes:", phonemes)
    print("len(phonemes):", len(phonemes))
    phoneme_ids = phonemizer.transform(phonemes)
    print("phoneme_ids:", phoneme_ids)
    print("len(phoneme_ids):", len(phoneme_ids))
