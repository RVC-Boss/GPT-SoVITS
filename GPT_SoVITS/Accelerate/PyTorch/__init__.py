import importlib.util
import os

import torch

from .sample_funcs import sample_naive
from .structs import T2SRequest, T2SResult
from .t2s_engine import T2SEngine as T2SEngineTorch

torch.set_grad_enabled(False)

if torch.__version__ >= "2.9.0":
    torch.backends.fp32_precision = "tf32"  # type: ignore
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    torch.backends.cuda.preferred_blas_library("cublaslt")


torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_accumulation = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

cpu_count = os.cpu_count() or 1
torch.set_num_threads(cpu_count)
torch.set_num_interop_threads(cpu_count)

backends = ["torch_varlen"]
if torch.cuda.is_available() and torch.version.cuda is not None:
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

BLACKWELL = False
if torch.cuda.is_available() and torch.version.cuda is not None:
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        sm_version = major + minor / 10.0
        if sm_version >= 9.0:
            BLACKWELL = True

quantization_methods_torch: list[str | None] = [None]
if importlib.util.find_spec("torchao") is not None:
    quantization_methods_torch.append("Int8")
    if BLACKWELL:
        quantization_methods_torch.append("FP8")
if BLACKWELL:
    quantization_methods_torch.append("FP8_E4M3FN")


__all__ = ["T2SEngineTorch", "T2SRequest", "sample_naive", "T2SResult", "backends", "quantization_methods_torch"]
