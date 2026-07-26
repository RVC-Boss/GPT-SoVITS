#!/usr/bin/env python3
"""Write sitecustomize.py that preloads CUDA12 NPP for torchcodec/torchaudio."""
from __future__ import annotations

import site
import sys
from pathlib import Path

SITECUSTOMIZE = r'''"""Auto-configure NVIDIA NPP libs for torchcodec/torchaudio."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path


def _npp_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia.npp")
        if spec is not None and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                if not loc:
                    continue
                p = Path(loc) / "lib"
                if p.is_dir():
                    dirs.append(p)
    except Exception:
        pass

    try:
        here = Path(__file__).resolve().parent / "nvidia" / "npp" / "lib"
        if here.is_dir():
            dirs.append(here)
    except Exception:
        pass
    return dirs


def _ensure_npp() -> None:
    for npp_lib in _npp_lib_dirs():
        lib_path = str(npp_lib)
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current.split(":") if p]
        if lib_path not in parts:
            os.environ["LD_LIBRARY_PATH"] = lib_path + ((":" + current) if current else "")
        for name in sorted(npp_lib.glob("libnpp*.so.12")):
            try:
                ctypes.CDLL(str(name), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        return


try:
    _ensure_npp()
except Exception:
    pass
'''


def main() -> int:
    candidates: list[Path] = []
    try:
        candidates.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    venv = Path(sys.prefix)
    candidates.append(venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")

    written = False
    for sp in candidates:
        if not sp.is_dir():
            continue
        if "site-packages" not in sp.parts:
            continue
        target = sp / "sitecustomize.py"
        target.write_text(SITECUSTOMIZE, encoding="utf-8")
        print(f"[OK] wrote {target}")
        written = True
        break

    if not written:
        print("[ERROR] could not find site-packages", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
