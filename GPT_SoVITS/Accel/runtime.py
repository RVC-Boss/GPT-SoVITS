from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACCEL_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ACCEL_ROOT / "vendor"
DLL_HANDLES = []

os.environ.setdefault("G2PW", "0")
if os.name == "nt":
    for directory in (
        VENDOR_ROOT / "cuda" / "bin",
        Path(sys.executable).resolve().parent / "Lib" / "site-packages" / "torch" / "lib",
    ):
        if directory.is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(directory)))
