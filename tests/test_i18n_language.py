from tools.i18n.i18n import resolve_language


def test_resolve_language_prefers_supported_last_cli_locale():
    assert resolve_language("fr_FR", ["webui.py", "zh_CN"]) == "zh_CN"


def test_resolve_language_keeps_language_for_unknown_last_argument():
    assert resolve_language("fr_FR", ["webui.py", "--port", "9872"]) == "fr_FR"


def test_resolve_language_keeps_language_for_empty_argv():
    assert resolve_language("fr_FR", []) == "fr_FR"
