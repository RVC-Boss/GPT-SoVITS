"""Optional PyTorch acceleration for the inference service."""

import os

from tools.portable_runtime import _prepare_dlls


os.environ.setdefault("G2PW", "0")
_prepare_dlls()
