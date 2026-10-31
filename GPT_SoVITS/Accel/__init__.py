"""Optional PyTorch acceleration for the inference service."""

import os

from GPT_SoVITS.runtime.portable import _prepare_dlls


os.environ.setdefault("G2PW", "0")
_prepare_dlls()
