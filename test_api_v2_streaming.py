"""GPU-free API v2 compatibility and semantic-streaming regression tests."""
import asyncio
import importlib.util
import inspect
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient


def _module(name, package=False, **attributes):
    result = types.ModuleType(name)
    if package:
        result.__path__ = []
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class _FakeConfig:
    def __init__(self, _path):
        self.languages = {"zh", "en"}
        self.version = "v2ProPlus"


class _FakePipeline:
    def __init__(self, config):
        self.configs = config
        self.calls = []
        self.script_factory = None
        self.closed = []
        self.setter_calls = []

    def run(self, request):
        self.calls.append(dict(request))
        if self.script_factory is not None:
            return self.script_factory(request)

        def generate():
            try:
                yield 32000, np.array([100, -100], dtype=np.int16)
                yield 32000, np.array([200, -200], dtype=np.int16)
            finally:
                self.closed.append(request.get("text"))

        return generate()

    def set_ref_audio(self, path):
        self.setter_calls.append(("ref", path, threading.current_thread().name))

    def init_t2s_weights(self, path):
        self.setter_calls.append(("gpt", path, threading.current_thread().name))

    def init_vits_weights(self, path):
        self.setter_calls.append(("sovits", path, threading.current_thread().name))


def _load_api_module():
    root = Path(__file__).resolve().parent
    portable = _module(
        "tools.portable_runtime",
        FFMPEG_EXE="ffmpeg",
        activate=lambda **_kwargs: None,
    )
    i18n = _module("tools.i18n.i18n", I18nAuto=lambda: object())
    tts = _module("GPT_SoVITS.TTS_infer_pack.TTS", TTS=_FakePipeline, TTS_Config=_FakeConfig)
    segmentation = _module(
        "GPT_SoVITS.TTS_infer_pack.text_segmentation_method",
        get_method_names=lambda: [f"cut{index}" for index in range(6)],
    )
    streaming_name = "GPT_SoVITS.TTS_infer_pack.streaming"
    streaming_spec = importlib.util.spec_from_file_location(
        streaming_name, root / "GPT_SoVITS" / "TTS_infer_pack" / "streaming.py"
    )
    streaming = importlib.util.module_from_spec(streaming_spec)
    replacements = {
        "tools": _module("tools", package=True),
        "tools.portable_runtime": portable,
        "tools.i18n": _module("tools.i18n", package=True),
        "tools.i18n.i18n": i18n,
        "tools.acceleration": _module("tools.acceleration", create_acceleration=lambda *_args, **_kwargs: None),
        "torch": _module("torch", inference_mode=lambda: lambda function: function),
        "GPT_SoVITS": _module("GPT_SoVITS", package=True),
        "GPT_SoVITS.TTS_infer_pack": _module("GPT_SoVITS.TTS_infer_pack", package=True),
        "GPT_SoVITS.TTS_infer_pack.TTS": tts,
        "GPT_SoVITS.TTS_infer_pack.text_segmentation_method": segmentation,
        streaming_name: streaming,
    }
    spec = importlib.util.spec_from_file_location("api_v2_streaming_test_module", root / "api_v2.py")
    api = importlib.util.module_from_spec(spec)
    replacements[spec.name] = api
    with patch.dict(sys.modules, replacements), patch.object(sys, "argv", [sys.argv[0]]):
        streaming_spec.loader.exec_module(streaming)
        spec.loader.exec_module(api)
    return api


class APIV2StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = _load_api_module()

    def setUp(self):
        self.api.tts_config.version = "v2ProPlus"
        self.api.tts_pipeline = _FakePipeline(self.api.tts_config)
        self.client = TestClient(self.api.APP)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.assertFalse(self.api._tts_pipeline_lock.locked())

    def request(self, **updates):
        request = {
            "text": "你好，今天开心吗？",
            "text_lang": "zh",
            "ref_audio_path": "reference.wav",
            "prompt_lang": "zh",
            "prompt_text": "你好。",
        }
        request.update(updates)
        return request

    def test_existing_defaults_and_routes_are_preserved(self):
        post_defaults = {
            "text": None,
            "text_lang": None,
            "ref_audio_path": None,
            "aux_ref_audio_paths": None,
            "prompt_lang": None,
            "prompt_text": "",
            "top_k": 15,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "use_cuda_graph": True,
            "repetition_penalty": 1.35,
            "sample_steps": None,
            "cfg_rate": None,
            "super_sampling": False,
            "overlap_length": 2,
            "min_chunk_length": 16,
        }
        fields = getattr(self.api.TTS_Request, "model_fields", None)
        if fields is None:
            fields = self.api.TTS_Request.__fields__
        for name, expected in post_defaults.items():
            self.assertEqual(fields[name].default, expected, name)

        get_defaults = {
            "text": None,
            "text_lang": None,
            "ref_audio_path": None,
            "aux_ref_audio_paths": None,
            "prompt_lang": None,
            "prompt_text": "",
            "top_k": 15,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "parallel_infer": True,
            "use_cuda_graph": True,
            "repetition_penalty": 1.35,
            "sample_steps": None,
            "cfg_rate": None,
            "super_sampling": False,
            "streaming_mode": False,
            "overlap_length": 2,
            "min_chunk_length": 16,
        }
        signature = inspect.signature(self.api.tts_get_endpoint)
        for name, expected in get_defaults.items():
            self.assertEqual(signature.parameters[name].default, expected, name)

        new_defaults = {
            "gpt_streaming": False,
            "streaming_chunk_seconds": 2.0,
            "streaming_window_mode": "fixed",
            "streaming_lookahead_seconds": 0.12,
            "streaming_overlap_seconds": 0.04,
            "gpt_streaming_lookahead_seconds": 0.5,
            "use_flash_attention": True,
        }
        for name, expected in new_defaults.items():
            self.assertEqual(fields[name].default, expected, name)
            self.assertEqual(signature.parameters[name].default, expected, name)

        routes = {}
        for route in self.api.APP.routes:
            if hasattr(route, "methods"):
                routes.setdefault(route.path, set()).update(route.methods)
        self.assertEqual(routes["/tts"], {"GET", "POST"})
        self.assertEqual(routes["/control"], {"GET"})
        self.assertEqual(routes["/set_refer_audio"], {"GET"})
        self.assertEqual(routes["/set_gpt_weights"], {"GET"})
        self.assertEqual(routes["/set_sovits_weights"], {"GET"})

    def test_get_and_post_pass_the_same_new_options(self):
        request = self.request(
            text_split_method="cut0",
            streaming_mode=2,
            gpt_streaming=True,
            streaming_chunk_seconds=1.5,
            streaming_lookahead_seconds=0.1,
            streaming_overlap_seconds=0.03,
            gpt_streaming_lookahead_seconds=0.6,
            use_flash_attention=False,
        )
        post_response = self.client.post("/tts", json=request)
        self.assertEqual(post_response.status_code, 200, post_response.text)
        post_call = self.api.tts_pipeline.calls[-1]

        get_response = self.client.get("/tts", params=request)
        self.assertEqual(get_response.status_code, 200, get_response.text)
        get_call = self.api.tts_pipeline.calls[-1]
        self.assertEqual(get_call, post_call)
        self.assertEqual(get_call["streaming_mode"], True)
        self.assertEqual(get_call["streaming_window_mode"], "fixed")
        self.assertFalse(get_call["fixed_length_chunk"])
        self.assertFalse(get_call["use_flash_attention"])

    def test_hybrid_window_mode_is_exposed_and_validated(self):
        self.api.tts_pipeline.configs.version = "v5turbo"
        request = self.request(
            text_split_method="cut0",
            streaming_mode=2,
            gpt_streaming=True,
            streaming_window_mode="hybrid",
        )
        post_response = self.client.post("/tts", json=request)
        self.assertEqual(post_response.status_code, 200, post_response.text)
        self.assertEqual(self.api.tts_pipeline.calls[-1]["streaming_window_mode"], "hybrid")

        get_response = self.client.get("/tts", params=request)
        self.assertEqual(get_response.status_code, 200, get_response.text)
        self.assertEqual(self.api.tts_pipeline.calls[-1], self.api.tts_pipeline.calls[-2])

        invalid_mode = dict(request, streaming_window_mode="unknown")
        self.assertEqual(self.client.post("/tts", json=invalid_mode).status_code, 400)
        self.api.tts_pipeline.configs.version = "v2ProPlus"
        v2_request = dict(request, text="V2 hybrid", prompt_text="参考")
        v2_response = self.client.post("/tts", json=v2_request)
        self.assertEqual(v2_response.status_code, 200, v2_response.text)
        self.assertEqual(self.api.tts_pipeline.calls[-1]["streaming_window_mode"], "hybrid")

        disabled = dict(v2_request, gpt_streaming=False)
        response = self.client.post("/tts", json=disabled)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(self.api.tts_pipeline.calls[-1]["gpt_streaming"])
        self.assertEqual(self.api.tts_pipeline.calls[-1]["streaming_window_mode"], "hybrid")

    def test_all_text_splitters_work_with_both_windows_over_get_and_post(self):
        for version in ("v5turbo", "v5dev", "v2ProPlus"):
            self.api.tts_pipeline.configs.version = version
            for method in (f"cut{i}" for i in range(6)):
                for window in ("fixed", "hybrid"):
                    with self.subTest(version=version, method=method, window=window):
                        request = self.request(text_split_method=method, streaming_mode=2,
                                               gpt_streaming=True, streaming_window_mode=window)
                        post = self.client.post("/tts", json=request)
                        self.assertEqual(post.status_code, 200, post.text)
                        get = self.client.get("/tts", params=request)
                        self.assertEqual(get.status_code, 200, get.text)
                        self.assertEqual(self.api.tts_pipeline.calls[-1], self.api.tts_pipeline.calls[-2])
                        self.assertEqual(self.api.tts_pipeline.calls[-1]["text_split_method"], method)

    def test_old_streaming_modes_keep_their_mapping(self):
        expected = {
            0: (False, False, False),
            1: (False, True, False),
            2: (True, False, False),
            3: (True, False, True),
            False: (False, False, False),
            True: (False, True, False),
        }
        for mode, mapping in expected.items():
            response = self.client.post("/tts", json=self.request(streaming_mode=mode))
            self.assertEqual(response.status_code, 200, response.text)
            call = self.api.tts_pipeline.calls[-1]
            self.assertEqual(
                (call["streaming_mode"], call["return_fragment"], call["fixed_length_chunk"]),
                mapping,
            )

    def test_api_validator_sees_original_mode_and_rejects_invalid_options(self):
        valid = self.request(text_split_method="cut0", streaming_mode=3, gpt_streaming=True)
        response = self.client.post("/tts", json=valid)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.api.tts_pipeline.calls[-1]["fixed_length_chunk"], True)

        before = len(self.api.tts_pipeline.calls)
        for bad in (
            {"streaming_mode": True},
            {"streaming_mode": 1},
            {"streaming_mode": 4},
            {"streaming_mode": 2.0},
            {"streaming_mode": "2"},
            {"streaming_mode": 2, "text_split_method": "not-a-method"},
            {"streaming_mode": 2, "media_type": "ogg"},
            {"streaming_mode": 2, "streaming_chunk_seconds": 0},
        ):
            invalid = self.request(text_split_method="cut0", streaming_mode=2, gpt_streaming=True)
            invalid.update(bad)
            response = self.client.post("/tts", json=invalid)
            self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.api.tts_pipeline.calls), before)

        invalid_get = self.request(text_split_method="cut0", streaming_mode="2.0", gpt_streaming=True)
        response = self.client.get("/tts", params=invalid_get)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.api.tts_pipeline.calls), before)

        self.api.tts_pipeline.configs.version = "v4"
        invalid_version = self.request(text_split_method="cut0", streaming_mode=2, gpt_streaming=True)
        response = self.client.post("/tts", json=invalid_version)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.api.tts_pipeline.calls), before)

    def test_streamed_wav_contains_one_header_and_empty_first_chunks_are_skipped(self):
        def generate(_request):
            yield 32000, np.array([], dtype=np.int16)
            yield 32000, np.array([100, -100], dtype=np.int16)
            yield 32000, np.array([200, -200], dtype=np.int16)

        self.api.tts_pipeline.script_factory = generate
        response = self.client.post(
            "/tts",
            json=self.request(text_split_method="cut0", streaming_mode=2, gpt_streaming=True),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content[:4], b"RIFF")
        self.assertEqual(response.content.count(b"RIFF"), 1)
        self.assertEqual(len(response.content), 44 + 8)
        self.assertEqual(
            response.content[44:],
            np.array([100, -100, 200, -200], dtype=np.int16).tobytes(),
        )

    def test_raw_stream_has_no_container_header(self):
        response = self.client.post(
            "/tts",
            json=self.request(
                text_split_method="cut0",
                streaming_mode=2,
                gpt_streaming=True,
                media_type="raw",
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, np.array([100, -100, 200, -200], dtype=np.int16).tobytes())

    def test_no_valid_first_pcm_returns_400_and_releases_lock(self):
        self.api.tts_pipeline.script_factory = lambda _request: iter(
            [(32000, np.array([], dtype=np.int16))]
        )
        response = self.client.post(
            "/tts",
            json=self.request(text_split_method="cut0", streaming_mode=2, gpt_streaming=True),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("no PCM audio", response.json()["Exception"])
        self.assertFalse(self.api._tts_pipeline_lock.locked())


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = _load_api_module()

    async def test_asgi_disconnect_closes_generator_before_next_request(self):
        pipeline = _FakePipeline(self.api.tts_config)
        self.api.tts_pipeline = pipeline
        entered_blocking_next = threading.Event()
        continue_generator = threading.Event()
        closed = threading.Event()

        def generate(request):
            try:
                yield 32000, np.array([1, 2], dtype=np.int16)
                yield 32000, np.array([3, 4], dtype=np.int16)
                entered_blocking_next.set()
                continue_generator.wait(timeout=5)
                yield 32000, np.array([5, 6], dtype=np.int16)
            finally:
                closed.set()
                pipeline.closed.append(request.get("text"))

        pipeline.script_factory = generate
        request = {
            "text": "取消后继续",
            "text_lang": "zh",
            "ref_audio_path": "reference.wav",
            "prompt_lang": "zh",
            "prompt_text": "参考",
            "text_split_method": "cut0",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": 2,
            "gpt_streaming": True,
        }
        response = await self.api.tts_handle(dict(request))
        messages = []

        async def receive_disconnect():
            reached_next = await asyncio.to_thread(entered_blocking_next.wait, 2)
            if not reached_next:
                continue_generator.set()
                self.fail("stream did not reach the blocking next() call")
            threading.Timer(0.1, continue_generator.set).start()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/tts",
            "raw_path": b"/tts",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "root_path": "",
        }
        try:
            await response(scope, receive_disconnect, send)
        finally:
            continue_generator.set()

        self.assertTrue(closed.is_set())
        self.assertTrue(any(message.get("type") == "http.response.start" for message in messages))
        self.assertFalse(self.api._tts_pipeline_lock.locked())

        pipeline.script_factory = None
        next_response = await self.api.tts_handle(dict(request))
        chunks = [chunk async for chunk in next_response.body_iterator]
        self.assertTrue(chunks)
        self.assertFalse(self.api._tts_pipeline_lock.locked())

    async def test_model_switch_waits_for_the_active_stream(self):
        self.api.tts_config.version = "v2ProPlus"
        pipeline = _FakePipeline(self.api.tts_config)
        self.api.tts_pipeline = pipeline
        request = {
            "text": "边播边切模型",
            "text_lang": "zh",
            "ref_audio_path": "reference.wav",
            "prompt_lang": "zh",
            "prompt_text": "参考",
            "text_split_method": "cut0",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": 2,
            "gpt_streaming": True,
        }
        response = await self.api.tts_handle(dict(request))
        setter = asyncio.create_task(self.api.set_gpt_weights("weights.pth"))
        await asyncio.sleep(0.02)
        self.assertEqual(pipeline.setter_calls, [])

        chunks = [chunk async for chunk in response.body_iterator]
        result = await setter
        self.assertTrue(chunks)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(pipeline.setter_calls[0][:2], ("gpt", "weights.pth"))
        self.assertNotEqual(pipeline.setter_calls[0][2], "MainThread")
        self.assertFalse(self.api._tts_pipeline_lock.locked())


if __name__ == "__main__":
    unittest.main()
