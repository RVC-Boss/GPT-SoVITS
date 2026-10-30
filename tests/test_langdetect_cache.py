"""Language detection caches stay writable when model mounts are read-only."""

import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "GPT_SoVITS" / "text" / "LangSegmenter" / "langsegmenter.py"


class LanguageDetectionCacheTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".codex" / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="langdetect-cache-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)

    def load_config(self, configured_cache=""):
        class Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Detector:
            def __init__(self, config):
                self.config = config

        infer = types.SimpleNamespace(LangDetectConfig=Config, LangDetector=Detector)
        fast_langdetect = types.ModuleType("fast_langdetect")
        fast_langdetect.infer = infer
        jieba = types.ModuleType("jieba")
        jieba.setLogLevel = lambda _level: None
        split_lang = types.ModuleType("split_lang")
        split_lang.LangSplitter = object
        namespace = {"__file__": str(self.project / "GPT_SoVITS/text/LangSegmenter/langsegmenter.py")}
        with patch.dict("sys.modules", {
            "fast_langdetect": fast_langdetect,
            "jieba": jieba,
            "split_lang": split_lang,
        }), patch.dict(os.environ, {"GPT_SOVITS_LANGDETECT_CACHE": configured_cache}):
            exec(compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec"), namespace)
        return infer._default_detector.config

    def test_default_cache_is_created_outside_model_directory(self):
        config = self.load_config()
        expected = self.project / ".codex/runtime-cache/fast_langdetect"
        self.assertEqual(Path(config.cache_dir), expected)
        self.assertTrue(expected.is_dir())
        self.assertIsNone(config.custom_model_path)
        self.assertFalse((self.project / "GPT_SoVITS/pretrained_models").exists())

    def test_environment_override_is_created_and_used(self):
        expected = self.project / "custom cache" / "language"
        config = self.load_config(str(expected))
        self.assertEqual(Path(config.cache_dir), expected)
        self.assertTrue(expected.is_dir())
        self.assertFalse((self.project / ".codex/runtime-cache").exists())

    def test_existing_model_is_used_without_copying_to_cache(self):
        legacy = self.project / "GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"existing model")
        config = self.load_config()
        self.assertEqual(Path(config.custom_model_path), legacy)
        self.assertEqual(legacy.read_bytes(), b"existing model")
        self.assertEqual(list(Path(config.cache_dir).iterdir()), [])

    def test_preseeded_cache_takes_priority_over_existing_model(self):
        legacy = self.project / "GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"existing model")
        cache = self.project / ".codex/runtime-cache/fast_langdetect"
        cache.mkdir(parents=True)
        (cache / "lid.176.bin").write_bytes(b"preseeded model")
        config = self.load_config()
        self.assertIsNone(config.custom_model_path)
        self.assertEqual(Path(config.cache_dir), cache)


if __name__ == "__main__":
    unittest.main()
