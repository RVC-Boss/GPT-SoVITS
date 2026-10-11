# -*- coding: utf-8 -*-
"""Russian phoneme helpers.

This module contains the checkpoint-facing normalization used to convert
Epitran-style XSAMPA symbols into the compact Latin inventory expected by
Russian GPT-SoVITS checkpoints.
"""

from __future__ import annotations


_XSAMPA_BASE_MAP = {
    "1": "I",
    "@": "A",
    "6": "A",
    "s`": "S",
    "S`": "S",
}
_ALLOWED_BASES = frozenset("ABDEFGIJKLMNOPRSTUVXZ")


def normalize_xsampa_symbol(symbol: str) -> str:
    """Normalize one Russian XSAMPA symbol to the checkpoint base inventory.

    Epitran emits multi-character and modified symbols for Russian, including
    palatalized consonants (for example ``d'``), long vowels, and ``s``` for
    Russian ``ш`` (IPA ``ʂ``). Russian checkpoints use a collapsed uppercase
    base inventory, so supported modifiers are removed after resolving known
    multi-character symbols.

    Raises:
        ValueError: if the normalized symbol is outside the supported Russian
            checkpoint inventory.
    """

    if not isinstance(symbol, str) or not symbol:
        raise ValueError("Russian XSAMPA symbol must be a non-empty string")

    normalized = _XSAMPA_BASE_MAP.get(symbol, symbol)
    normalized = normalized.replace("'", "").replace(":", "").replace("\\", "")
    normalized = _XSAMPA_BASE_MAP.get(normalized, normalized).upper()

    if normalized not in _ALLOWED_BASES:
        raise ValueError(f"Unsupported Russian XSAMPA symbol: {symbol!r}")
    return normalized
