from pathlib import Path
from typing import Callable

from torch import Tensor


def walk_paths(root, suffix):
    for path in Path(root).iterdir():
        if path.is_dir():
            yield from walk_paths(path, suffix)
        elif path.suffix == suffix:
            yield path


def rglob_audio_files(path: Path):
    return list(walk_paths(path, ".wav")) + list(walk_paths(path, ".flac"))


def mix_fg_bg(fg: Tensor, bg: Tensor, alpha: float | Callable[..., float] = 0.5, eps=1e-7):
    """
    Args:
        fg: (b, t)
        bg: (b, t)
    """
    assert bg.shape == fg.shape, f"bg.shape != fg.shape: {bg.shape} != {fg.shape}"
    fg = fg / (fg.abs().max(dim=-1, keepdim=True).values + eps)
    bg = bg / (bg.abs().max(dim=-1, keepdim=True).values + eps)

    fg_energy = fg.pow(2).sum(dim=-1, keepdim=True)
    bg_energy = bg.pow(2).sum(dim=-1, keepdim=True)

    fg = fg / (fg_energy + eps).sqrt()
    bg = bg / (bg_energy + eps).sqrt()

    if callable(alpha):
        alpha = alpha()

    assert 0 <= alpha <= 1, f"alpha must be between 0 and 1: {alpha}"

    mx = alpha * fg + (1 - alpha) * bg
    mx = mx / (mx.abs().max(dim=-1, keepdim=True).values + eps)

    return mx
