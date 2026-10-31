"""Runtime initialization for the bundled GPT-SoVITS environment."""

from __future__ import annotations

import os
import shutil
import sys
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from types import ModuleType

from .paths import CACHE_DIR, PACKAGE_ROOT, PROJECT_ROOT, RUNTIME_DIR, SITE_PACKAGES_DIR


# Backward-compatible name used by the former tools.portable_runtime module.
SITE_PACKAGES = SITE_PACKAGES_DIR


def _resolve_binary(bundled_name: str, system_name: str) -> str:
    bundled = RUNTIME_DIR / bundled_name
    if bundled.exists():
        return str(bundled)
    return shutil.which(system_name) or system_name


FFMPEG_EXE = _resolve_binary("ffmpeg.exe", "ffmpeg")
FFPROBE_EXE = _resolve_binary("ffprobe.exe", "ffprobe")
_DLL_HANDLES = []
_DLL_READY = False
_ACTIVE = False
_FLASH_IMPORT_READY = False
_FLASH_IMPORT_LOCK = RLock()


def _prepare_dlls() -> None:
    global _DLL_READY
    if _DLL_READY or os.name != "nt":
        return
    package_directories = [SITE_PACKAGES_DIR, Path(sys.prefix) / "Lib" / "site-packages"]
    directories = [
        RUNTIME_DIR,
        *(packages / "torch" / "lib" for packages in package_directories),
        PACKAGE_ROOT / "Accel" / "vendor" / "cuda" / "bin",
        *(directory for packages in package_directories for directory in sorted((packages / "nvidia").glob("*/bin"))),
    ]
    for directory in dict.fromkeys(directories):
        if directory.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            # cuDNN dynamically loads its component DLLs through the process PATH.
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    _DLL_READY = True


def _lazy_apply_rotary_emb(*args, **kwargs):
    return import_module("flash_attn.layers.rotary").apply_rotary_emb(*args, **kwargs)


def prepare_transformers_flash_attention() -> None:
    global _FLASH_IMPORT_READY
    with _FLASH_IMPORT_LOCK:
        if _FLASH_IMPORT_READY:
            return
        _prepare_dlls()
        import_utils = import_module("transformers.utils.import_utils")
        if import_utils.is_flash_attn_2_available():
            from GPT_SoVITS.Accel.capabilities import _flash_attention_module

            if _flash_attention_module() is None:
                import_utils.is_flash_attn_2_available = lambda: False
                import_module("transformers.utils").is_flash_attn_2_available = import_utils.is_flash_attn_2_available
        if os.name != "nt" or version("transformers") != "4.51.3":
            _FLASH_IMPORT_READY = True
            return
        # Transformers 4.51.3 imports Triton rotary even for BERT/HuBERT, which never use it.
        rotary_name = "flash_attn.layers.rotary"
        previous_rotary = sys.modules.get(rotary_name)
        if previous_rotary is None:
            rotary = ModuleType(rotary_name)
            rotary.__spec__ = ModuleSpec(rotary_name, loader=None)
            rotary.apply_rotary_emb = _lazy_apply_rotary_emb
            sys.modules[rotary_name] = rotary
        try:
            import_module("transformers.modeling_flash_attention_utils")
        finally:
            if previous_rotary is None:
                sys.modules.pop(rotary_name, None)
        _FLASH_IMPORT_READY = True


def activate(*, change_cwd: bool = False) -> None:
    global _ACTIVE
    if change_cwd:
        os.chdir(PROJECT_ROOT)
    if not _ACTIVE:
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PACKAGE_ROOT))
        if RUNTIME_DIR.is_dir():
            os.environ["PATH"] = str(RUNTIME_DIR) + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("G2PW", "0")
        nltk_data = RUNTIME_DIR / "nltk_data"
        if not nltk_data.is_dir():
            nltk_data = CACHE_DIR / "nltk_data"
        if nltk_data.is_dir():
            os.environ.setdefault("NLTK_DATA", str(nltk_data))
        try:
            import _distutils_hack

            _distutils_hack.add_shim()
        except ImportError:
            pass
        _prepare_dlls()
        _ACTIVE = True
    prepare_transformers_flash_attention()
