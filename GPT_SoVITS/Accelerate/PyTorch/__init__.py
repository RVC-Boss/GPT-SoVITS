import importlib.util

import torch

from .sample_funcs import sample_naive
from .structs import T2SRequest, T2SResult
from .t2s_engine import T2SEngine as T2SEngineTorch

torch.set_grad_enabled(False)

backends = ["torch_varlen"]
if torch.cuda.is_available():
    backends.append("torch_static_cuda_graph")
    # if importlib.util.find_spec("sageattention") is not None:
    #     for i in range(torch.cuda.device_count()):
    #         major, minor = torch.cuda.get_device_capability(i)
    #         sm_version = major + minor / 10.0
    #         if sm_version >= 7.0:
    #             backends.append("sage_attn_varlen_cuda_graph")
    if importlib.util.find_spec("flash_attn") is not None:
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            sm_version = major + minor / 10.0
            if sm_version >= 7.5:
                backends.append("flash_attn_varlen_cuda_graph")
# if torch.mps.is_available():
#     backends.append("mps_flash_attn_varlen")


__all__ = ["T2SEngineTorch", "T2SRequest", "sample_naive", "T2SResult", "backends"]
