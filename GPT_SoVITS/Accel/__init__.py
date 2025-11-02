from . import MLX, PyTorch
from .logger import console, logger, tb
from .MLX import quantization_methods_mlx
from .PyTorch import T2SEngineTorch, T2SRequest, T2SResult, quantization_methods_torch
from .PyTorch.AR.structs import T2SEngineProtocol

backends = PyTorch.backends + MLX.backends

backends = [
    b.replace("_", "-").title().replace("Mlx", "MLX").replace("Mps", "MPS").replace("Cuda", "CUDA") for b in backends
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
    "quantization_methods_torch",
    "quantization_methods_mlx",
]
