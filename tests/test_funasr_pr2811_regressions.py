from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_funasr_module(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    class FakeAutoModel:
        def __new__(cls, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(generate=lambda **_kwargs: [{"text": "ok"}])

    funasr = ModuleType("funasr")
    funasr.AutoModel = FakeAutoModel
    modelscope = ModuleType("modelscope")
    modelscope.snapshot_download = lambda *args, **kwargs: None
    tqdm = ModuleType("tqdm")
    tqdm.tqdm = lambda values: values
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)

    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm)
    monkeypatch.setitem(sys.modules, "torch", torch)

    module_path = REPO_ROOT / "tools" / "asr" / "funasr_asr.py"
    spec = importlib.util.spec_from_file_location("funasr_asr_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


@pytest.mark.parametrize(
    ("backend", "expected_model"),
    [
        ("sensevoice", "iic/SenseVoiceSmall"),
        ("fun-asr-nano", "FunAudioLLM/Fun-ASR-Nano-2512"),
    ],
)
def test_yue_honors_the_selected_multilingual_backend(
    monkeypatch: pytest.MonkeyPatch, backend: str, expected_model: str
) -> None:
    module, calls = _load_funasr_module(monkeypatch)

    module.create_model("yue", backend=backend)

    assert calls[-1]["model"] == expected_model


def test_nvidia_npp_requirement_does_not_break_macos() -> None:
    requirement_line = next(
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("nvidia-npp-cu12")
    )
    requirement = Requirement(requirement_line)

    assert requirement.marker is not None
    assert not requirement.marker.evaluate(
        {
            "sys_platform": "darwin",
            "platform_machine": "arm64",
            "python_version": "3.11",
        }
    )


def test_cpu_uv_setup_does_not_install_cuda_npp(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_log = tmp_path / "uv.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$*" >> "$UV_LOG"
            if [[ "${1:-}" == "venv" ]]; then
              venv="${@: -1}"
              mkdir -p "$venv/bin" "$venv/nltk_data"
              printf 'export PATH="%s/bin:$PATH"\n' "$venv" > "$venv/bin/activate"
              printf '%s\n' \
                '#!/usr/bin/env bash' \
                'if [[ "${1:-}" == "-V" ]]; then echo "Python 3.10.0"; fi' \
                'if [[ "${1:-}" == "-c" && "${2:-}" == *"sys.prefix"* ]]; then cd "$(dirname "$0")/.." && pwd; fi' \
                'exit 0' > "$venv/bin/python"
              chmod +x "$venv/bin/python"
            fi
            """
        ),
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["UV_LOG"] = str(uv_log)
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "setup_uv.sh"),
            "--device",
            "CPU",
            "--skip-models",
            "--venv",
            str(tmp_path / "venv"),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "nvidia-npp-cu12" not in uv_log.read_text(encoding="utf-8")
