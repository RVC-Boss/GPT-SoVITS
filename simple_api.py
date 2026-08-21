"""
Small profile-based API layer for GPT-SoVITS.

Run:
    python simple_api.py -c simple_api.yaml

Then call:
    POST http://127.0.0.1:9881/speak
    {"text": "hello", "voice": "default"}
"""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
import wave
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import (  # noqa: E402
    get_method_names as get_cut_method_names,
)


DEFAULT_TTS_PARAMS: Dict[str, Any] = {
    "text": "",
    "text_lang": "zh",
    "ref_audio_path": "",
    "aux_ref_audio_paths": [],
    "prompt_text": "",
    "prompt_lang": "zh",
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "text_split_method": "cut5",
    "batch_size": 1,
    "batch_threshold": 0.75,
    "split_bucket": True,
    "speed_factor": 1.0,
    "fragment_interval": 0.3,
    "seed": -1,
    "media_type": "wav",
    "streaming_mode": False,
    "return_fragment": False,
    "fixed_length_chunk": False,
    "parallel_infer": True,
    "repetition_penalty": 1.35,
    "sample_steps": 32,
    "super_sampling": False,
    "overlap_length": 2,
    "min_chunk_length": 16,
}

SUPPORTED_MEDIA_TYPES = {"wav", "raw", "ogg", "aac"}
SUPPORTED_UPLOAD_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}
VOICE_META_KEYS = {"description"}


class SpeakRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    text_lang: Optional[str] = None
    ref_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[List[str]] = None
    prompt_text: Optional[str] = None
    prompt_lang: Optional[str] = None
    format: Optional[str] = Field(default=None, description="wav, raw, ogg, or aac")
    stream: Optional[bool] = Field(default=None, description="true maps to streaming mode 2")
    streaming_mode: Optional[Union[bool, int]] = Field(default=None, description="0, 1, 2, 3, true, or false")
    speed: Optional[float] = Field(default=None, description="Alias of speed_factor")
    speed_factor: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    text_split_method: Optional[str] = None
    batch_size: Optional[int] = None
    batch_threshold: Optional[float] = None
    split_bucket: Optional[bool] = None
    fragment_interval: Optional[float] = None
    seed: Optional[int] = None
    parallel_infer: Optional[bool] = None
    repetition_penalty: Optional[float] = None
    sample_steps: Optional[int] = None
    super_sampling: Optional[bool] = None
    overlap_length: Optional[int] = None
    min_chunk_length: Optional[int] = None


class WeightsRequest(BaseModel):
    gpt_weights_path: Optional[str] = None
    sovits_weights_path: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-SoVITS simple API")
    parser.add_argument("-c", "--config", default="simple_api.yaml", help="simple API config path")
    parser.add_argument("--tts-config", default=None, help="GPT-SoVITS tts_infer.yaml path")
    parser.add_argument("-a", "--bind-addr", default=None, help="bind address")
    parser.add_argument("-p", "--port", type=int, default=None, help="bind port")
    return parser.parse_args()


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"simple API config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("simple API config must be a YAML object")
    return data


args = parse_args()
simple_config = load_yaml_config(args.config)
server_config = simple_config.get("server", {}) or {}
host = args.bind_addr or server_config.get("host", "127.0.0.1")
port = args.port or int(server_config.get("port", 9881))
tts_config_path = args.tts_config or server_config.get("tts_config", "GPT_SoVITS/configs/tts_infer.yaml")

cut_method_names = get_cut_method_names()
tts_config: Optional[TTS_Config] = None
tts_pipeline: Optional[TTS] = None
infer_lock = threading.Lock()

APP = FastAPI(
    title="GPT-SoVITS Simple API",
    description=(
        "简化接口层，封装 GPT-SoVITS 推理引擎。\n\n"
        "## 核心流程\n"
        "1. 上传 3-10 秒参考音频（或视频，前端自动提取音频）\n"
        "2. 填写需要生成的文字\n"
        "3. 调用 `/api/tts` 获取生成的音频\n\n"
        "## 其他接口\n"
        "- `/speak` — 基于 voice profile 的调用方式\n"
        "- `/v1/tts` — OpenAI 兼容格式\n"
        "- `/admin/*` — 管理接口（热加载配置、切换模型）"
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "MVP", "description": "核心 TTS 接口，上传参考音频直接生成"},
        {"name": "Profile", "description": "基于 voice profile 的调用方式"},
        {"name": "Admin", "description": "管理接口：热加载配置、切换模型权重"},
        {"name": "System", "description": "健康检查与信息查询"},
    ],
)
APP.add_middleware(
    CORSMiddleware,
    allow_origins=simple_config.get("cors_allow_origins", ["*"]),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
test_frontend_dir = PROJECT_ROOT / "test_frontend"
if test_frontend_dir.exists():
    APP.mount("/test", StaticFiles(directory=str(test_frontend_dir), html=True), name="test_frontend")


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    def handle_pack_ogg() -> None:
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)

    try:
        threading.stack_size(4096 * 4096)
        pack_thread = threading.Thread(target=handle_pack_ogg)
        pack_thread.start()
        pack_thread.join()
    except (RuntimeError, ValueError):
        handle_pack_ogg()
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    del rate
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    del io_buffer
    wav_buffer = BytesIO()
    sf.write(wav_buffer, data, rate, format="wav")
    return wav_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-vn",
            "-f",
            "adts",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str) -> BytesIO:
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def wave_header_chunk(frame_input: bytes = b"", channels: int = 1, sample_width: int = 2, sample_rate: int = 32000) -> bytes:
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


def request_to_dict(request: BaseModel) -> Dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump(exclude_none=True)
    return request.dict(exclude_none=True)


def get_default_voice_name() -> Optional[str]:
    default_voice = simple_config.get("default_voice")
    if default_voice:
        return str(default_voice)
    voices = simple_config.get("voices", {}) or {}
    if "default" in voices:
        return "default"
    return next(iter(voices.keys()), None)


def resolve_project_path(path_value: Optional[str]) -> Optional[str]:
    if path_value in [None, ""]:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def normalize_streaming(streaming_mode: Optional[Union[bool, int]], stream: Optional[bool]) -> Tuple[bool, bool, bool, bool]:
    if streaming_mode is None:
        streaming_mode = 2 if stream else 0
    elif isinstance(streaming_mode, bool):
        streaming_mode = 2 if streaming_mode else 0

    if streaming_mode == 0:
        return False, False, False, False
    if streaming_mode == 1:
        return False, True, False, True
    if streaming_mode == 2:
        return True, False, False, True
    if streaming_mode == 3:
        return True, False, True, True
    raise HTTPException(status_code=400, detail="streaming_mode must be 0, 1, 2, 3, true, or false")


def get_upload_config() -> Dict[str, Any]:
    return simple_config.get("upload", {}) or {}


def get_upload_root() -> Path:
    upload_dir = str(get_upload_config().get("dir", "runtime/uploads"))
    root = Path(upload_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_request_upload_dir() -> Path:
    request_dir = get_upload_root() / uuid.uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def get_upload_limits() -> Tuple[float, float, int]:
    upload_config = get_upload_config()
    min_seconds = float(upload_config.get("min_ref_seconds", 3.0))
    max_seconds = float(upload_config.get("max_ref_seconds", 10.0))
    max_upload_mb = int(upload_config.get("max_upload_mb", 80))
    return min_seconds, max_seconds, max_upload_mb


def get_upload_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix:
        return suffix
    content_type = (upload.content_type or "").lower()
    if content_type in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return ".wav"
    if content_type == "audio/mpeg":
        return ".mp3"
    if content_type == "audio/ogg":
        return ".ogg"
    if content_type in {"audio/aac", "audio/aacp"}:
        return ".aac"
    return ".wav"


def validate_audio_upload(upload: UploadFile) -> None:
    suffix = get_upload_suffix(upload)
    content_type = (upload.content_type or "").lower()
    is_audio_type = content_type.startswith("audio/") or content_type in {"application/octet-stream", ""}
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS or not is_audio_type:
        raise HTTPException(status_code=400, detail=f"unsupported audio upload: {upload.filename or content_type}")


async def save_upload_file(upload: UploadFile, target_dir: Path, name_prefix: str) -> str:
    validate_audio_upload(upload)
    _, _, max_upload_mb = get_upload_limits()
    max_bytes = max_upload_mb * 1024 * 1024
    suffix = get_upload_suffix(upload)
    target_path = target_dir / f"{name_prefix}{suffix}"
    total_size = 0

    try:
        with target_path.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_bytes:
                    raise HTTPException(status_code=400, detail=f"audio upload exceeds {max_upload_mb} MB")
                f.write(chunk)
        if total_size == 0:
            raise HTTPException(status_code=400, detail=f"audio upload is empty: {upload.filename or name_prefix}")
    finally:
        await upload.close()

    return str(target_path)


def get_audio_duration_with_ffprobe(audio_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    return float(result.stdout.strip())


def get_audio_duration_seconds(audio_path: str) -> float:
    try:
        info = sf.info(audio_path)
        if info.samplerate <= 0:
            raise ValueError("invalid sample rate")
        return float(info.frames) / float(info.samplerate)
    except Exception as sf_exc:
        try:
            return get_audio_duration_with_ffprobe(audio_path)
        except Exception as ffprobe_exc:
            raise HTTPException(
                status_code=400,
                detail=f"unable to read audio duration: {audio_path}. soundfile={sf_exc}; ffprobe={ffprobe_exc}",
            ) from ffprobe_exc


def validate_main_ref_duration(audio_path: str) -> None:
    min_seconds, max_seconds, _ = get_upload_limits()
    duration = get_audio_duration_seconds(audio_path)
    if duration < min_seconds or duration > max_seconds:
        raise HTTPException(
            status_code=400,
            detail=f"ref_audio duration must be between {min_seconds:g}s and {max_seconds:g}s, got {duration:.2f}s",
        )


def apply_emotion_preset(payload: Dict[str, Any], emotion: Optional[str]) -> None:
    if emotion in [None, ""]:
        return
    emotion_name = str(emotion).strip().lower()
    emotion_presets = simple_config.get("emotion_presets", {}) or {}
    if emotion_name not in emotion_presets:
        supported = ", ".join(sorted(emotion_presets.keys())) or "none"
        raise HTTPException(status_code=400, detail=f"emotion is not configured: {emotion_name}. supported: {supported}")
    payload.update(emotion_presets[emotion_name] or {})


def prompt_text_required_for_current_model() -> bool:
    if tts_config is None:
        return False
    use_vocoder = getattr(tts_config, "use_vocoder", None)
    if use_vocoder is not None:
        return bool(use_vocoder)
    return getattr(tts_config, "version", None) in {"v3", "v4"}


def voice_public_info(name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    ref_audio_path = resolve_project_path(profile.get("ref_audio_path"))
    return {
        "name": name,
        "description": profile.get("description", ""),
        "text_lang": profile.get("text_lang"),
        "prompt_lang": profile.get("prompt_lang"),
        "ref_audio_path": profile.get("ref_audio_path"),
        "ready": bool(ref_audio_path and Path(ref_audio_path).exists() and profile.get("prompt_lang")),
    }


def build_tts_request(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str, bool]:
    voices = simple_config.get("voices", {}) or {}
    explicit_ref_audio = bool(payload.get("ref_audio_path"))
    voice_name = payload.pop("voice", None)
    if voice_name is None and not explicit_ref_audio:
        voice_name = get_default_voice_name()
    voice_profile: Dict[str, Any] = {}

    if voice_name:
        if voice_name not in voices:
            raise HTTPException(status_code=404, detail=f"voice not found: {voice_name}")
        voice_profile = deepcopy(voices[voice_name] or {})

    tts_req = deepcopy(DEFAULT_TTS_PARAMS)
    tts_req.update(simple_config.get("defaults", {}) or {})
    tts_req.update({k: v for k, v in voice_profile.items() if k not in VOICE_META_KEYS})

    stream = payload.pop("stream", None)
    media_type = payload.pop("format", None)
    speed = payload.pop("speed", None)
    if media_type is not None:
        tts_req["media_type"] = media_type

    for key, value in payload.items():
        if key in DEFAULT_TTS_PARAMS:
            tts_req[key] = value
    if speed is not None:
        tts_req["speed_factor"] = speed

    ref_audio_path = resolve_project_path(tts_req.get("ref_audio_path"))
    if ref_audio_path:
        tts_req["ref_audio_path"] = ref_audio_path

    aux_paths = tts_req.get("aux_ref_audio_paths") or []
    tts_req["aux_ref_audio_paths"] = [resolve_project_path(item) for item in aux_paths if item]

    media_type = str(tts_req.get("media_type", "wav")).lower()
    tts_req["media_type"] = media_type
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise HTTPException(status_code=400, detail=f"format is not supported: {media_type}")

    text = str(tts_req.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    tts_req["text"] = text

    if not tts_req.get("ref_audio_path"):
        raise HTTPException(status_code=400, detail="ref_audio_path is required in voice profile or request")
    if not Path(tts_req["ref_audio_path"]).exists():
        raise HTTPException(status_code=400, detail=f"ref_audio_path does not exist: {tts_req['ref_audio_path']}")

    for aux_path in tts_req["aux_ref_audio_paths"]:
        if aux_path and not Path(aux_path).exists():
            raise HTTPException(status_code=400, detail=f"aux_ref_audio_path does not exist: {aux_path}")

    if tts_config is None:
        raise HTTPException(status_code=503, detail="TTS pipeline is not ready")

    text_lang = str(tts_req.get("text_lang") or "").lower()
    prompt_lang = str(tts_req.get("prompt_lang") or "").lower()
    tts_req["text_lang"] = text_lang
    tts_req["prompt_lang"] = prompt_lang

    if text_lang not in tts_config.languages:
        raise HTTPException(status_code=400, detail=f"text_lang is not supported: {text_lang}")
    if prompt_lang not in tts_config.languages:
        raise HTTPException(status_code=400, detail=f"prompt_lang is not supported: {prompt_lang}")
    if not str(tts_req.get("prompt_text") or "").strip() and prompt_text_required_for_current_model():
        version = getattr(tts_config, "version", "current")
        raise HTTPException(status_code=400, detail=f"prompt_text is required when using GPT-SoVITS {version}")

    split_method = str(tts_req.get("text_split_method") or "cut5")
    if split_method not in cut_method_names:
        raise HTTPException(status_code=400, detail=f"text_split_method is not supported: {split_method}")

    streaming_mode = tts_req.pop("streaming_mode", None)
    streaming_enabled, return_fragment, fixed_length_chunk, response_stream = normalize_streaming(streaming_mode, stream)
    tts_req["streaming_mode"] = streaming_enabled
    tts_req["return_fragment"] = return_fragment
    tts_req["fixed_length_chunk"] = fixed_length_chunk

    return tts_req, media_type, response_stream


def synthesize_once(payload: Dict[str, Any]) -> Tuple[bytes, str]:
    tts_req, media_type, response_stream = build_tts_request(payload)
    if response_stream:
        raise HTTPException(status_code=400, detail="base64 output does not support streaming")

    if tts_pipeline is None:
        raise HTTPException(status_code=503, detail="TTS pipeline is not ready")

    try:
        with infer_lock:
            sr, audio_data = next(tts_pipeline.run(tts_req))
        audio_bytes = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
        return audio_bytes, media_type
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"message": "tts failed", "exception": str(exc)}) from exc


def synthesize_response(payload: Dict[str, Any]) -> Response:
    tts_req, media_type, response_stream = build_tts_request(payload)
    if tts_pipeline is None:
        raise HTTPException(status_code=503, detail="TTS pipeline is not ready")

    if response_stream:

        def streaming_generator() -> Generator[bytes, None, None]:
            first_chunk = True
            chunk_media_type = media_type
            with infer_lock:
                for sr, chunk in tts_pipeline.run(tts_req):
                    if first_chunk and chunk_media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        chunk_media_type = "raw"
                        first_chunk = False
                    yield pack_audio(BytesIO(), chunk, sr, chunk_media_type).getvalue()

        return StreamingResponse(streaming_generator(), media_type=f"audio/{media_type}")

    try:
        with infer_lock:
            sr, audio_data = next(tts_pipeline.run(tts_req))
        audio_bytes = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
        return Response(audio_bytes, media_type=f"audio/{media_type}")
    except Exception as exc:
        return JSONResponse(status_code=400, content={"message": "tts failed", "exception": str(exc)})


@APP.on_event("startup")
def startup() -> None:
    global tts_config, tts_pipeline
    tts_config = TTS_Config(tts_config_path)
    print(tts_config)
    tts_pipeline = TTS(tts_config)


@APP.get("/", tags=["System"])
def index() -> Dict[str, Any]:
    return {
        "name": "GPT-SoVITS Simple API",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": {
            "system": ["/health", "/voices"],
            "mvp": ["/api/tts"],
            "profile": ["/speak", "/speak/base64", "/v1/tts"],
            "admin": ["/admin/reload-config", "/admin/weights"],
        },
    }


@APP.get("/health", tags=["System"], summary="健康检查")
def health() -> Dict[str, Any]:
    import os

    result: Dict[str, Any] = {
        "status": "ok" if tts_pipeline is not None else "starting",
        "tts_config": tts_config_path,
        "version": getattr(tts_config, "version", None),
        "languages": getattr(tts_config, "languages", []),
        "pid": os.getpid(),
    }
    try:
        import psutil
        result["memory_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_used_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
                "memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024, 1),
            }
    except Exception:
        pass
    return result


@APP.get("/voices", tags=["System"], summary="列出可用 voice profiles")
def list_voices() -> Dict[str, Any]:
    voices = simple_config.get("voices", {}) or {}
    return {
        "default_voice": get_default_voice_name(),
        "voices": [voice_public_info(name, profile or {}) for name, profile in voices.items()],
    }


@APP.post(
    "/api/tts",
    tags=["MVP"],
    summary="核心 TTS 接口",
    description=(
        "上传参考音频和需要生成的文字，返回生成的音频。\n\n"
        "**主参考音频要求**：3-10 秒，支持 wav/flac/ogg/mp3/m4a/aac 格式。\n\n"
        "**文字切句**：固定使用 `cut5`（按标点符号切句）。\n\n"
        "**情绪预设**：neutral / happy / calm / sad / angry，本质是映射到采样和语速参数。"
    ),
)
async def mvp_tts(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    aux_ref_audio: Optional[List[UploadFile]] = File(default=None),
    prompt_text: str = Form(default=""),
    text_lang: str = Form(default="zh"),
    prompt_lang: str = Form(default="zh"),
    format: str = Form(default="wav"),
    emotion: Optional[str] = Form(default=None),
    speed: Optional[float] = Form(default=None),
    seed: int = Form(default=-1),
) -> Response:
    request_dir = get_request_upload_dir()
    try:
        ref_audio_path = await save_upload_file(ref_audio, request_dir, "ref")
        validate_main_ref_duration(ref_audio_path)

        aux_ref_audio_paths = []
        for index, upload in enumerate(aux_ref_audio or []):
            aux_path = await save_upload_file(upload, request_dir, f"aux_{index}")
            get_audio_duration_seconds(aux_path)
            aux_ref_audio_paths.append(aux_path)

        payload: Dict[str, Any] = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "prompt_text": prompt_text or "",
            "prompt_lang": prompt_lang,
            "format": format,
            "text_split_method": "cut5",
            "streaming_mode": 0,
            "seed": seed,
        }
        apply_emotion_preset(payload, emotion)
        payload["text_split_method"] = "cut5"
        payload["streaming_mode"] = 0
        if speed is not None:
            payload["speed"] = speed

        return synthesize_response(payload)
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


@APP.get("/speak", tags=["Profile"], summary="GET 方式调用 voice profile TTS")
def speak_get(
    text: str,
    voice: Optional[str] = None,
    text_lang: Optional[str] = None,
    format: Optional[str] = None,
    stream: Optional[bool] = None,
    speed: Optional[float] = None,
) -> Response:
    payload = {
        "text": text,
        "voice": voice,
        "text_lang": text_lang,
        "format": format,
        "stream": stream,
        "speed": speed,
    }
    return synthesize_response({k: v for k, v in payload.items() if v is not None})


@APP.post("/speak", tags=["Profile"], summary="POST 方式调用 voice profile TTS")
def speak_post(request: SpeakRequest) -> Response:
    return synthesize_response(request_to_dict(request))


@APP.post("/v1/tts", tags=["Profile"], summary="OpenAI 兼容格式 TTS")
def openai_style_tts(request: SpeakRequest) -> Response:
    return synthesize_response(request_to_dict(request))


@APP.post("/speak/base64", tags=["Profile"], summary="返回 Base64 编码的音频")
def speak_base64(request: SpeakRequest) -> Dict[str, Any]:
    audio_bytes, media_type = synthesize_once(request_to_dict(request))
    return {
        "media_type": f"audio/{media_type}",
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
    }


@APP.post("/admin/reload-config", tags=["Admin"], summary="热加载 simple_api.yaml 配置")
def reload_config() -> Dict[str, Any]:
    global simple_config
    simple_config = load_yaml_config(args.config)
    return {"message": "success", "default_voice": get_default_voice_name()}


@APP.post("/admin/weights", tags=["Admin"], summary="运行时切换 GPT-SoVITS 模型权重")
def set_weights(request: WeightsRequest) -> Dict[str, Any]:
    if tts_pipeline is None:
        raise HTTPException(status_code=503, detail="TTS pipeline is not ready")
    if not request.gpt_weights_path and not request.sovits_weights_path:
        raise HTTPException(status_code=400, detail="gpt_weights_path or sovits_weights_path is required")

    try:
        with infer_lock:
            if request.gpt_weights_path:
                tts_pipeline.init_t2s_weights(request.gpt_weights_path)
            if request.sovits_weights_path:
                tts_pipeline.init_vits_weights(request.sovits_weights_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"message": "change weights failed", "exception": str(exc)}) from exc

    return {"message": "success"}


if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(app=APP, host=None if host == "None" else host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
