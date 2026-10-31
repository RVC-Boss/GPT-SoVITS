"""Checkpoint loading helpers used by the inference service."""

import hashlib
import os
from io import BytesIO

import torch


_HEAD_TO_VERSION = {
    b"00": ("v1", "v1", False),
    b"01": ("v2", "v2", False),
    b"02": ("v2", "v3", False),
    b"03": ("v2", "v3", True),
    b"04": ("v2", "v4", True),
    b"05": ("v2", "v2Pro", False),
    b"06": ("v2", "v2ProPlus", False),
    b"07": ("v2", "v5dev", True),
    b"08": ("v2", "v5turbo", True),
}

_PRETRAINED_HASH_TO_VERSION = {
    "dc3c97e17592963677a4a1681f30c653": ("v2", "v2", False),
    "43797be674a37c1c83ee81081941ed0f": ("v2", "v3", False),
    "6642b37f3dbb1f76882b69937c95a5f3": ("v2", "v2", False),
    "4f26b9476d0c5033e04162c486074374": ("v2", "v4", False),
    "c7e9fce2223f3db685cdfa1e6368728a": ("v2", "v2Pro", False),
    "66b313e39455b57ab1b0bc0b239c9d0a": ("v2", "v2ProPlus", False),
    "c4ce1f839a0d271e41963f9fb16ade76": ("v2", "v5dev", False),
    "6bc5cb2a195cc8ec22b2e723cd4d6f47": ("v2", "v5turbo", False),
}


def get_hash_from_file(path: str) -> str:
    """Return the MD5 of the first 8 KiB used for pretrained identification."""

    with open(path, "rb") as stream:
        return hashlib.md5(stream.read(8192)).hexdigest()


def get_sovits_version_from_path_fast(path: str):
    """Identify a SoVITS checkpoint without loading its full state dict."""

    pretrained_version = _PRETRAINED_HASH_TO_VERSION.get(get_hash_from_file(path))
    if pretrained_version is not None:
        return pretrained_version

    with open(path, "rb") as stream:
        header = stream.read(2)
    if header != b"PK":
        try:
            return _HEAD_TO_VERSION[header]
        except KeyError as exc:
            raise ValueError(f"Unsupported SoVITS checkpoint header: {header!r}") from exc

    size = os.path.getsize(path)
    if size < 82978 * 1024:
        return "v1", "v1", False
    if size < 700 * 1024 * 1024:
        return "v2", "v2", False
    return "v2", "v3", False


def load_sovits_new(path: str):
    """Load both legacy and ZIP-based checkpoints into a CPU state dict."""

    with open(path, "rb") as stream:
        header = stream.read(2)
        if header != b"PK":
            payload = b"PK" + stream.read()
            buffer = BytesIO(payload)
            buffer.seek(0)
            return torch.load(buffer, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)
