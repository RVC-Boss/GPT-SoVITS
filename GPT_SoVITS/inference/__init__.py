"""Canonical public inference API."""

from GPT_SoVITS.runtime.portable import activate as _activate

_activate(change_cwd=False)

from .config import TTSConfig, TTS_Config
from .pipeline import TTS

__all__ = ["TTS", "TTSConfig", "TTS_Config"]
