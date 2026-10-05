"""Select optional AR acceleration for the requested device and precision."""

from functools import lru_cache
from importlib import import_module
import logging
import os
from pathlib import Path
import sys

import torch


_logger = logging.getLogger(__name__)
_DLL_HANDLES = []


def _cuda_available(device):
    return device.type == "cuda" and torch.cuda.is_available() and torch.version.cuda is not None


def cuda_graph_available(device):
    device = torch.device(device)
    return bool(
        _cuda_available(device)
        and os.environ.get("CUDAGraph", "1") != "0"
        and callable(getattr(torch.cuda, "CUDAGraph", None))
        and callable(getattr(torch.cuda, "graph", None))
        and callable(getattr(torch.cuda, "Stream", None))
        and callable(getattr(torch.nn.functional, "scaled_dot_product_attention", None))
    )


@lru_cache(maxsize=1)
def _flash_attention_module():
    if os.name == "nt":
        root = Path(__file__).resolve().parents[1]
        for directory in (
            root / "GPT_SoVITS" / "Accel" / "vendor" / "cuda" / "bin",
            Path(sys.executable).resolve().parent / "Lib" / "site-packages" / "torch" / "lib",
        ):
            if directory.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    try:
        return import_module("flash_attn")
    except (ImportError, OSError, RuntimeError) as exc:
        _logger.info("FlashAttention is unavailable: %s", exc)
        return None


def flash_attention_available(device, dtype):
    device = torch.device(device)
    if not _cuda_available(device) or dtype not in (torch.float16, torch.bfloat16):
        return False
    if torch.cuda.get_device_capability(device)[0] < 8:
        return False
    module = _flash_attention_module()
    return callable(getattr(module, "flash_attn_with_kvcache", None))


def resolve_acceleration(device, dtype, use_cuda_graph=True, use_flash_attention=True):
    graph = bool(use_cuda_graph and cuda_graph_available(device))
    if use_flash_attention and flash_attention_available(device, dtype):
        return "flash_attn_varlen_cuda_graph", graph
    if graph:
        return "torch_static_cuda_graph", True
    return None, False


def create_acceleration(
    weights_path,
    device,
    dtype,
    max_batch_size=32,
    use_cuda_graph=True,
    use_flash_attention=True,
):
    backend, _ = resolve_acceleration(device, dtype, use_cuda_graph, use_flash_attention)
    if backend is None:
        return None
    try:
        from GPT_SoVITS.Accel.adapter import AccelInference
        import_module("GPT_SoVITS.Accel.PyTorch.AR.backends." + backend)
    except (ImportError, OSError, RuntimeError) as exc:
        _logger.warning("AR acceleration is unavailable; using ordinary inference: %s", exc)
        return None
    return AccelInference(
        weights_path,
        device=device,
        dtype=dtype,
        max_batch_size=max_batch_size,
        use_cuda_graph=use_cuda_graph,
        use_flash_attention=use_flash_attention,
    )
