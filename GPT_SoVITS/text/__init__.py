import os
# if os.environ.get("version","v1")=="v1":
#   from text.symbols import symbols
# else:
#   from text.symbols2 import symbols

from text import symbols as symbols_v1
from text import symbols2 as symbols_v2

_symbol_to_id_v1 = {s: i for i, s in enumerate(symbols_v1.symbols)}
_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2.symbols)}


def cleaned_text_to_sequence(cleaned_text, version=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if version is None:
        version = os.environ.get("version", "v2")
    symbol_to_id = _symbol_to_id_v1 if version == "v1" else _symbol_to_id_v2
    unk_id = symbol_to_id.get("UNK")
    phones = []
    for symbol in cleaned_text:
        if symbol in symbol_to_id:
            phones.append(symbol_to_id[symbol])
        elif unk_id is not None:
            phones.append(unk_id)

    return phones
