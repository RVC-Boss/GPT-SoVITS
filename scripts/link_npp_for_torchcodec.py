#!/usr/bin/env python3
"""Symlink CUDA12 NPP shared libs next to torchcodec so dlopen can find them."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import torchcodec
    except ImportError:
        print("[SKIP] torchcodec not installed")
        return 0

    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia.npp")
        if spec is None or not spec.submodule_search_locations:
            print("[SKIP] nvidia.npp not installed")
            return 0
        npp_roots = [Path(p) for p in spec.submodule_search_locations if p]
    except Exception as exc:
        print(f"[SKIP] nvidia.npp lookup failed: {exc}")
        return 0

    npp_lib = None
    for root in npp_roots:
        cand = root / "lib"
        if cand.is_dir():
            npp_lib = cand
            break
    if npp_lib is None:
        print("[SKIP] nvidia.npp lib dir not found")
        return 0

    torchcodec_dir = Path(torchcodec.__file__).resolve().parent
    linked = 0
    for src in sorted(npp_lib.glob("libnpp*.so.12")):
        dst = torchcodec_dir / src.name
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() and dst.resolve() == src.resolve():
                continue
            dst.unlink()
        dst.symlink_to(src)
        linked += 1
        print(f"[OK] {dst.name} -> {src}")

    print(f"[OK] linked {linked} NPP libs into {torchcodec_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
