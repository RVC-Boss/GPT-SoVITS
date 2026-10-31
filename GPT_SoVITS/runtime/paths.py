"""Canonical filesystem locations used by the inference runtime."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
SITE_PACKAGES_DIR = RUNTIME_DIR / "Lib" / "site-packages"
CACHE_DIR = PROJECT_ROOT / ".codex" / "runtime-cache"
PRETRAINED_MODELS_DIR = PACKAGE_ROOT / "pretrained_models"

G2PW_MODEL_DIR = PACKAGE_ROOT / "text" / "G2PWModel"
BERT_MODEL_DIR = PRETRAINED_MODELS_DIR / "chinese-roberta-wwm-ext-large"
CNHUBERT_MODEL_DIR = PRETRAINED_MODELS_DIR / "chinese-hubert-base"
BIGVGAN_VOCODER_DIR = PRETRAINED_MODELS_DIR / "models--nvidia--bigvgan_v2_24khz_100band_256x"
SPEAKER_MODEL_PATH = PRETRAINED_MODELS_DIR / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"

AP_BWE_DIR = PACKAGE_ROOT / "vendor" / "ap_bwe"
AP_BWE_CHECKPOINT_DIR = AP_BWE_DIR / "24kto48k"
AP_BWE_CHECKPOINT_PATH = AP_BWE_CHECKPOINT_DIR / "g_24kto48k.zip"
LEGACY_AP_BWE_CHECKPOINT_DIR = PROJECT_ROOT / "tools" / "AP_BWE_main" / "24kto48k"
LEGACY_AP_BWE_CHECKPOINT_PATH = LEGACY_AP_BWE_CHECKPOINT_DIR / "g_24kto48k.zip"


def package_path(*parts: str, project_root: str | Path = PROJECT_ROOT) -> Path:
    """Build a path below the GPT_SoVITS package for an alternate project root."""

    return Path(project_root) / "GPT_SoVITS" / Path(*parts)


def pretrained_path(*parts: str, project_root: str | Path = PROJECT_ROOT) -> Path:
    """Build a path below the pretrained-model directory."""

    return package_path("pretrained_models", *parts, project_root=project_root)


def cache_path(name: str, *, project_root: str | Path = PROJECT_ROOT, env_var: str | None = None) -> Path:
    """Return a writable runtime-cache location, honoring an existing override."""

    configured = os.environ.get(env_var, "").strip() if env_var else ""
    if configured:
        return Path(configured).expanduser()
    return Path(project_root) / ".codex" / "runtime-cache" / name


def vocoder_checkpoint(version: str) -> Path:
    """Return the local V3/V4/V5 vocoder checkpoint for a model version."""

    return PRETRAINED_MODELS_DIR / f"gsv-{version}-pretrained" / "vocoder.pth"


def resolve_ap_bwe_checkpoint(checkpoint: str | Path | None = None) -> Path:
    """Resolve AP-BWE weights, preferring the vendored path over the legacy mount."""

    if checkpoint is not None:
        return Path(checkpoint).expanduser()
    if AP_BWE_CHECKPOINT_PATH.is_file():
        return AP_BWE_CHECKPOINT_PATH
    if LEGACY_AP_BWE_CHECKPOINT_PATH.is_file():
        return LEGACY_AP_BWE_CHECKPOINT_PATH
    return AP_BWE_CHECKPOINT_PATH
