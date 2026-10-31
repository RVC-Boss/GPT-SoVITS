"""Static checks for the inference-only repository surface."""

import ast
import unittest
from itertools import chain
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    return modules


class InferenceSurfaceTests(unittest.TestCase):
    def test_only_inference_entrypoint_remains(self):
        self.assertTrue((ROOT / "api_v2.py").is_file())
        for relative in (
            "api.py",
            "webui.py",
            "GPT_SoVITS/inference_webui.py",
            "GPT_SoVITS/inference_webui_fast.py",
            "GPT_SoVITS/inference_gui.py",
            "GPT_SoVITS/inference_cli.py",
            "GPT_SoVITS/s1_train.py",
            "GPT_SoVITS/s2_train.py",
            "GPT_SoVITS/s2_train_v3.py",
            "GPT_SoVITS/s2_train_v3_lora.py",
            "GPT_SoVITS/prepare_datasets",
            "tools/asr",
            "tools/uvr5",
        ):
            self.assertFalse((ROOT / relative).exists(), relative)


    def test_runtime_python_imports_do_not_reference_removed_services(self):
        forbidden = {
            "gradio",
            "funasr",
            "faster_whisper",
            "ctranslate2",
            "modelscope",
            "pytorch_lightning",
            "torchmetrics",
            "tensorboard",
        }
        offenders = []
        for path in chain(
            [ROOT / "api_v2.py"],
            (ROOT / "GPT_SoVITS").rglob("*.py"),
            (ROOT / "tools").rglob("*.py"),
        ):
            imported = _imported_modules(path)
            overlap = sorted(imported & forbidden)
            if overlap:
                offenders.append((path.relative_to(ROOT).as_posix(), overlap))
        self.assertEqual(offenders, [])


    def test_api_v2_exposes_inference_routes_only(self):
        source = (ROOT / "GPT_SoVITS" / "service" / "routes.py").read_text(encoding="utf-8")
        self.assertIn('application.add_api_route("/tts", tts_get_endpoint, methods=["GET"])', source)
        self.assertIn('application.add_api_route("/tts", tts_post_endpoint, methods=["POST"])', source)
        self.assertIn('application.add_api_route("/set_refer_audio", set_refer_aduio, methods=["GET"])', source)
        self.assertIn('application.add_api_route("/set_gpt_weights", set_gpt_weights, methods=["GET"])', source)
        self.assertIn('application.add_api_route("/set_sovits_weights", set_sovits_weights, methods=["GET"])', source)
        self.assertNotIn('application.add_api_route("/control"', source)
