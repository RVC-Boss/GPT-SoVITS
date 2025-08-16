import importlib.util

import torch

from .sample_funcs import sample_naive
from .structs import T2SRequest, T2SResult
from .t2s_engine import T2SEngine as T2SEngineTorch

backends = ["naive"]
if torch.cuda.is_available():
    if importlib.util.find_spec("sageattention") is not None:
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            sm_version = major + minor / 10.0
            if sm_version >= 7.0:
                backends.append("sage_attn")
    if importlib.util.find_spec("flash_attn") is not None:
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            sm_version = major + minor / 10.0
            if sm_version >= 7.5:
                backends.append("flash_attn")


__all__ = ["T2SEngineTorch", "T2SRequest", "sample_naive", "T2SResult", "backends"]
