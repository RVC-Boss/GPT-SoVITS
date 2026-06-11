import asyncio
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_API_PATH = PROJECT_ROOT / "simple_api.py"


class DummyHTTPException(Exception):
    def __init__(self, status_code=500, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class DummyFastAPI:
    def __init__(self, *args, **kwargs):
        self.routes = []

    def add_middleware(self, *args, **kwargs):
        pass

    def mount(self, path, app, name=None):
        self.routes.append(types.SimpleNamespace(path=path, endpoint=app, name=name))

    def get(self, path):
        return self._route(path)

    def post(self, path):
        return self._route(path)

    def on_event(self, event):
        return lambda func: func

    def _route(self, path):
        def decorator(func):
            self.routes.append(types.SimpleNamespace(path=path, endpoint=func))
            return func

        return decorator


class DummyBaseModel:
    def dict(self, exclude_none=False):
        data = dict(self.__dict__)
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data


class DummyTTSConfig:
    version = "v2"
    languages = ["zh", "en", "auto"]
    use_vocoder = False


class FakeUpload:
    def __init__(self, filename, chunks, content_type="audio/wav"):
        self.filename = filename
        self.content_type = content_type
        self._chunks = list(chunks)
        self.closed = False

    async def read(self, size):
        del size
        if self._chunks:
            return self._chunks.pop(0)
        return b""

    async def close(self):
        self.closed = True


def install_dummy_modules():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = DummyFastAPI
    fastapi.File = lambda default=None, **kwargs: default
    fastapi.Form = lambda default=None, **kwargs: default
    fastapi.HTTPException = DummyHTTPException
    fastapi.Response = type("Response", (), {})
    fastapi.UploadFile = type("UploadFile", (), {})
    sys.modules["fastapi"] = fastapi

    middleware = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = type("CORSMiddleware", (), {})
    sys.modules["fastapi.middleware"] = middleware
    sys.modules["fastapi.middleware.cors"] = cors

    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = type("JSONResponse", (), {})
    responses.StreamingResponse = type("StreamingResponse", (), {})
    sys.modules["fastapi.responses"] = responses

    staticfiles = types.ModuleType("fastapi.staticfiles")
    staticfiles.StaticFiles = type("StaticFiles", (), {"__init__": lambda self, *args, **kwargs: None})
    sys.modules["fastapi.staticfiles"] = staticfiles

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = DummyBaseModel
    pydantic.Field = lambda default=None, **kwargs: default
    sys.modules["pydantic"] = pydantic

    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    sys.modules["numpy"] = numpy

    soundfile = types.ModuleType("soundfile")
    soundfile.info = lambda path: types.SimpleNamespace(frames=16000 * 5, samplerate=16000)
    sys.modules["soundfile"] = soundfile

    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda stream: {
        "server": {"host": "127.0.0.1", "port": 9881, "tts_config": "dummy.yaml"},
        "upload": {"dir": "runtime/test_uploads", "min_ref_seconds": 3, "max_ref_seconds": 10, "max_upload_mb": 1},
        "defaults": {"text_lang": "zh", "prompt_lang": "zh", "media_type": "wav", "text_split_method": "cut5"},
        "emotion_presets": {"calm": {"temperature": 0.8, "speed_factor": 0.9}},
        "voices": {},
    }
    sys.modules["yaml"] = yaml

    sys.modules["GPT_SoVITS"] = types.ModuleType("GPT_SoVITS")
    sys.modules["GPT_SoVITS.TTS_infer_pack"] = types.ModuleType("GPT_SoVITS.TTS_infer_pack")

    tts_module = types.ModuleType("GPT_SoVITS.TTS_infer_pack.TTS")
    tts_module.TTS = type("DummyTTS", (), {})
    tts_module.TTS_Config = DummyTTSConfig
    sys.modules["GPT_SoVITS.TTS_infer_pack.TTS"] = tts_module

    segmentation = types.ModuleType("GPT_SoVITS.TTS_infer_pack.text_segmentation_method")
    segmentation.get_method_names = lambda: ["cut0", "cut5"]
    sys.modules["GPT_SoVITS.TTS_infer_pack.text_segmentation_method"] = segmentation


def load_simple_api():
    install_dummy_modules()
    sys.argv = ["simple_api.py"]
    spec = importlib.util.spec_from_file_location("simple_api_under_test", SIMPLE_API_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.tts_config = DummyTTSConfig()
    return module


class SimpleApiContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simple_api = load_simple_api()

    def test_api_tts_route_is_registered(self):
        routes = {route.path for route in self.simple_api.APP.routes}
        self.assertIn("/api/tts", routes)

    def test_build_request_accepts_explicit_ref_without_voice_profile(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as ref:
            req, media_type, response_stream = self.simple_api.build_tts_request(
                {
                    "text": "Hello, test.",
                    "ref_audio_path": ref.name,
                    "prompt_text": "",
                    "prompt_lang": "zh",
                    "text_lang": "zh",
                    "format": "wav",
                    "text_split_method": "cut5",
                    "streaming_mode": 0,
                }
            )

        self.assertEqual(req["prompt_text"], "")
        self.assertEqual(req["text_split_method"], "cut5")
        self.assertEqual(req["ref_audio_path"], ref.name)
        self.assertEqual(media_type, "wav")
        self.assertFalse(response_stream)

    def test_explicit_speed_overrides_emotion_speed_preset(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as ref:
            payload = {
                "text": "Hello.",
                "ref_audio_path": ref.name,
                "prompt_lang": "zh",
                "text_lang": "zh",
                "format": "wav",
                "streaming_mode": 0,
                "speed": 1.05,
            }
            self.simple_api.apply_emotion_preset(payload, "calm")
            payload["speed"] = 1.05
            req, _, _ = self.simple_api.build_tts_request(payload)

        self.assertEqual(req["temperature"], 0.8)
        self.assertEqual(req["speed_factor"], 1.05)

    def test_empty_prompt_text_is_rejected_for_vocoder_models(self):
        original_use_vocoder = self.simple_api.tts_config.use_vocoder
        original_version = self.simple_api.tts_config.version
        self.simple_api.tts_config.use_vocoder = True
        self.simple_api.tts_config.version = "v3"

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as ref:
                with self.assertRaises(DummyHTTPException) as exc:
                    self.simple_api.build_tts_request(
                        {
                            "text": "Hello.",
                            "ref_audio_path": ref.name,
                            "prompt_text": "",
                            "prompt_lang": "zh",
                            "text_lang": "zh",
                            "format": "wav",
                            "text_split_method": "cut5",
                            "streaming_mode": 0,
                        }
                    )
            self.assertEqual(exc.exception.status_code, 400)
            self.assertIn("prompt_text is required", str(exc.exception.detail))
        finally:
            self.simple_api.tts_config.use_vocoder = original_use_vocoder
            self.simple_api.tts_config.version = original_version

    def test_ref_audio_duration_limits(self):
        self.simple_api.sf.info = lambda path: types.SimpleNamespace(frames=16000 * 5, samplerate=16000)
        self.simple_api.validate_main_ref_duration("ok.wav")

        self.simple_api.sf.info = lambda path: types.SimpleNamespace(frames=16000 * 2, samplerate=16000)
        with self.assertRaises(DummyHTTPException) as too_short:
            self.simple_api.validate_main_ref_duration("short.wav")
        self.assertEqual(too_short.exception.status_code, 400)

        self.simple_api.sf.info = lambda path: types.SimpleNamespace(frames=16000 * 11, samplerate=16000)
        with self.assertRaises(DummyHTTPException) as too_long:
            self.simple_api.validate_main_ref_duration("long.wav")
        self.assertEqual(too_long.exception.status_code, 400)

    def test_save_upload_file_writes_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            upload = FakeUpload("ref.wav", [b"abc", b"def"])
            saved_path = asyncio.run(self.simple_api.save_upload_file(upload, Path(tmp), "ref"))

            self.assertTrue(upload.closed)
            self.assertEqual(Path(saved_path).read_bytes(), b"abcdef")

    def test_mvp_tts_builds_payload_from_uploads_and_cleans_tmp_dir(self):
        captured = {}

        with tempfile.TemporaryDirectory() as tmp:
            request_dir = Path(tmp) / "request"
            request_dir.mkdir()

            original_get_request_upload_dir = self.simple_api.get_request_upload_dir
            original_synthesize_response = self.simple_api.synthesize_response
            original_sf_info = self.simple_api.sf.info

            def fake_synthesize_response(payload):
                captured.update(payload)
                return "audio-response"

            try:
                self.simple_api.get_request_upload_dir = lambda: request_dir
                self.simple_api.sf.info = lambda path: types.SimpleNamespace(frames=16000 * 5, samplerate=16000)
                self.simple_api.synthesize_response = fake_synthesize_response

                response = asyncio.run(
                    self.simple_api.mvp_tts(
                        text="Hello, test.",
                        ref_audio=FakeUpload("ref.wav", [b"main"]),
                        aux_ref_audio=[FakeUpload("aux.wav", [b"aux"])],
                        prompt_text="",
                        text_lang="zh",
                        prompt_lang="zh",
                        format="wav",
                        emotion="calm",
                        speed=1.05,
                        seed=123,
                    )
                )
            finally:
                self.simple_api.get_request_upload_dir = original_get_request_upload_dir
                self.simple_api.synthesize_response = original_synthesize_response
                self.simple_api.sf.info = original_sf_info

            self.assertEqual(response, "audio-response")
            self.assertFalse(request_dir.exists())

        self.assertEqual(captured["text"], "Hello, test.")
        self.assertEqual(captured["prompt_text"], "")
        self.assertEqual(captured["text_lang"], "zh")
        self.assertEqual(captured["prompt_lang"], "zh")
        self.assertEqual(captured["format"], "wav")
        self.assertEqual(captured["text_split_method"], "cut5")
        self.assertEqual(captured["streaming_mode"], 0)
        self.assertEqual(captured["seed"], 123)
        self.assertEqual(captured["temperature"], 0.8)
        self.assertEqual(captured["speed"], 1.05)
        self.assertEqual(len(captured["aux_ref_audio_paths"]), 1)


if __name__ == "__main__":
    unittest.main()
