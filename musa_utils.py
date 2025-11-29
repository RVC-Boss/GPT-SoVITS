import torch
from typing import Optional
import re

_musa_available = False
_musa_err_msg = "torch_musa not found or not configured correctly."

try:
    if hasattr(torch, 'musa') and torch.musa.is_available():
        _musa_available = True
except Exception:
    _musa_available = False


def is_available() -> bool:
    return _musa_available

def get_device() -> Optional[torch.device]:
    return torch.device("musa") if _musa_available else None

def device_count() -> int:
    if _musa_available:
        try:
            return torch.musa.device_count()
        except Exception:
            return 0
    return 0

def set_device(device_index: int):
    if _musa_available:
        try:
            torch.musa.set_device(device_index)
        except Exception:
            pass

def empty_cache():
    if _musa_available:
        try:
            torch.musa.empty_cache()
        except Exception:
            pass

def manual_seed(seed: int):
    if _musa_available:
        try:
            torch.musa.manual_seed(seed)
        except Exception:
            pass

def manual_seed_all(seed: int):
    if _musa_available:
        try:
            torch.musa.manual_seed_all(seed)
        except Exception:
            pass

def get_device_dtype(device_idx: int = 0) -> tuple[torch.dtype, float, float]:
    properties = torch.musa.get_device_properties(device_idx)
    major, minor = torch.musa.get_device_capability()
    version_val = major + minor / 10.0
    mem_bytes = properties.total_memory
    mem_gb = mem_bytes / (1024**3) + 0.4
    device_name = properties.name
    dtype = torch.float16
    numbers_in_name = [int(n) for n in re.findall(r'\d+', device_name)]
    if version_val <= 2.1:
        dtype = torch.float32
    return dtype, version_val, mem_gb

def should_ddp(device_idx: int = 0) -> bool:
    _, version_val, _ = get_device_dtype(device_idx)
    if version_val <= 2.1:
        return False
    else:
        return True

DEVICE: Optional[torch.device] = get_device()
