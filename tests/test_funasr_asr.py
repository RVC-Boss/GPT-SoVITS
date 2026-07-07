import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "asr" / "funasr_asr.py"


def load_funasr_asr(monkeypatch, auto_model):
    funasr = types.ModuleType("funasr")
    funasr.AutoModel = auto_model
    monkeypatch.setitem(sys.modules, "funasr", funasr)

    modelscope = types.ModuleType("modelscope")
    modelscope.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)

    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable: iterable
    monkeypatch.setitem(sys.modules, "tqdm", tqdm)

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    module_name = "funasr_asr_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fun_asr_nano_falls_back_to_modelscope_when_hf_config_is_not_resolved(monkeypatch):
    calls = []

    def fake_auto_model(**kwargs):
        calls.append(kwargs.copy())
        if kwargs["hub"] == "hf":
            raise RuntimeError("model 'FunAudioLLM/Fun-ASR-Nano-2512' is not registered.")
        if kwargs["hub"] == "ms":
            return types.SimpleNamespace(hub=kwargs["hub"], model_name=kwargs["model"])
        raise AssertionError(f"unexpected hub {kwargs['hub']}")

    module = load_funasr_asr(monkeypatch, fake_auto_model)

    model = module.create_model("zh", backend="fun-asr-nano")

    assert model.hub == "ms"
    assert model.model_name == "FunAudioLLM/Fun-ASR-Nano-2512"
    assert [(call["model"], call["hub"], call["trust_remote_code"]) for call in calls] == [
        ("FunAudioLLM/Fun-ASR-Nano-2512", "hf", True),
        ("FunAudioLLM/Fun-ASR-Nano-2512", "ms", False),
    ]
    assert all(call["vad_model"] == "fsmn-vad" for call in calls)
    assert all(call["device"] == "cpu" for call in calls)
    assert all(call["disable_update"] is True for call in calls)


def test_fun_asr_nano_keeps_non_registration_errors_visible(monkeypatch):
    def fake_auto_model(**kwargs):
        raise RuntimeError("network unavailable")

    module = load_funasr_asr(monkeypatch, fake_auto_model)

    with pytest.raises(RuntimeError, match="network unavailable"):
        module.create_model("zh", backend="fun-asr-nano")
