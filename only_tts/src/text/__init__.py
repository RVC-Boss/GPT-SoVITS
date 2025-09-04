import os
# Only V2+ symbols are supported (V4 and V2Pro use V2+ language set)
from text import symbols2 as symbols_v2

_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2.symbols)}


def cleaned_text_to_sequence(cleaned_text, version=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    # Only V2+ versions are supported
    phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]
    return phones
