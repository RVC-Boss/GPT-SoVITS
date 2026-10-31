"""Construct optional AR acceleration adapters with ordinary-inference fallback."""

from importlib import import_module
import logging

from .capabilities import resolve_acceleration


_logger = logging.getLogger(__name__)


def create_acceleration(
    weights_path,
    device,
    dtype,
    max_batch_size=32,
    use_cuda_graph=True,
    use_flash_attention=True,
    max_seq_length=None,
):
    backend, _ = resolve_acceleration(device, dtype, use_cuda_graph, use_flash_attention)
    if backend is None:
        return None
    try:
        from .adapter import AccelInference

        # Preserve the dynamic backend module names consumed by T2SEngine.
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
        max_seq_length=max_seq_length,
    )
