from . import MLX, PyTorch
from .logger import console, logger, tb
from .PyTorch import T2SEngineTorch, T2SRequest, T2SResult
from .PyTorch.structs import T2SEngineProtocol

backends = PyTorch.backends + MLX.backends

backends = [
    b.replace("_", "-")
    .title()
    .replace("Mlx", "MLX")
    .replace("Mps", "MPS")
    .replace("Cuda", "CUDA")
    .replace("Mxfp4", "MXFP4")
    for b in backends
]


__all__ = [
    "T2SEngineTorch",
    "T2SRequest",
    "T2SResult",
    "backends",
    "MLX",
    "PyTorch",
    "logger",
    "console",
    "tb",
    "T2SEngineProtocol",
]
