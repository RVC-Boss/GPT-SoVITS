"""Compatibility shim for acceleration capabilities and construction."""

from GPT_SoVITS.Accel.capabilities import (  # noqa: F401
    _flash_attention_module,
    cuda_graph_available,
    flash_attention_available,
    resolve_acceleration,
)
from GPT_SoVITS.Accel.factory import create_acceleration  # noqa: F401

__all__ = [
    "create_acceleration",
    "cuda_graph_available",
    "flash_attention_available",
    "resolve_acceleration",
]
