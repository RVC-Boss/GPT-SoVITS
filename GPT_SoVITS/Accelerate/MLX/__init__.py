import importlib.util
import platform

if importlib.util.find_spec("mlx") is not None and platform.system() == "Darwin":
    from .sample_funcs_mlx import sample_naive as sample_naive_mlx
    from .t2s_engine_mlx import T2SEngine as T2SEngineMLX

    backends = ["mlx_static", "mlx_quantized_mxfp4", "mlx_quantized_affine", "mlx_varlen"]
else:
    backends = []

__all__ = ["T2SEngineMLX", "sample_naive_mlx", "backends"]
