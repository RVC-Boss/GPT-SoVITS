"""Compatibility shim for the canonical AP-BWE inference adapter."""

from GPT_SoVITS.inference.super_resolution import AP_BWE
from GPT_SoVITS.runtime.paths import LEGACY_AP_BWE_CHECKPOINT_DIR

AP_BWE_main_dir_path = str(LEGACY_AP_BWE_CHECKPOINT_DIR.parent)

__all__ = ["AP_BWE", "AP_BWE_main_dir_path"]
