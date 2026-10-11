import pytest

from GPT_SoVITS.text.russian import normalize_xsampa_symbol


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("d'", "D"),
        ("n'", "N"),
        ("r'", "R"),
        ("l'", "L"),
        ("a:", "A"),
        ("1", "I"),
        ("@", "A"),
        ("6", "A"),
        ("s`", "S"),
        ("S`", "S"),
    ],
)
def test_normalize_russian_xsampa_symbol(symbol, expected):
    assert normalize_xsampa_symbol(symbol) == expected


@pytest.mark.parametrize("symbol", ["", "?", "t_s", None])
def test_normalize_russian_xsampa_symbol_rejects_unsupported_input(symbol):
    with pytest.raises(ValueError, match="Russian XSAMPA|Unsupported Russian"):
        normalize_xsampa_symbol(symbol)
