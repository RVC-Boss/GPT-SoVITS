"""
# asr_api.py usage

` python asr_api.py -a 127.0.0.1 -p 9881 `

## 调用:

### 语音转文本

endpoint: `/asr`

GET:
`http://127.0.0.1:9881/asr?audio_path=data/voice_ref/ref.wav&language=zh`

POST:
```json
{
  "audio_path": "data/voice_ref/ref.wav",
  "language": "zh",
  "engine": "auto",
  "model_size": "large-v3",
  "precision": "float16"
}
```

RESP:
成功: {"text": "..."}
失败: {"message": "..."} (http code 400)
"""

import argparse
import os
import signal
import sys
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(f"{now_dir}/GPT_SoVITS")

from faster_whisper import WhisperModel

from tools.asr.config import get_models
from tools.asr.fasterwhisper_asr import download_model, language_code_list
from tools.asr.funasr_asr import only_asr
from tools.my_utils import load_cudnn


load_cudnn()

parser = argparse.ArgumentParser(description="GPT-SoVITS ASR api")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default=9881, help="default: 9881")
parser.add_argument(
    "-s",
    "--model_size",
    type=str,
    default="large-v3",
    choices=get_models(),
    help="default: large-v3",
)
parser.add_argument(
    "-l",
    "--language",
    type=str,
    default="auto",
    choices=language_code_list,
    help="default language",
)
parser.add_argument(
    "-pr",
    "--precision",
    type=str,
    default="float16",
    choices=["float16", "float32", "int8"],
    help="compute precision for faster-whisper",
)
args = parser.parse_args()

host = args.bind_addr
port = args.port
argv = sys.argv

APP = FastAPI()


class ASRRequest(BaseModel):
    audio_path: Optional[str] = None
    language: Optional[str] = None
    engine: str = "auto"  # auto | fasterwhisper | funasr
    model_size: Optional[str] = None
    precision: Optional[str] = None


whisper_models = {}
whisper_model_paths = {}


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    audio_path = req.get("audio_path")
    language = req.get("language", "auto")
    engine = req.get("engine", "auto")
    model_size = req.get("model_size", args.model_size)
    precision = req.get("precision", args.precision)

    if not audio_path:
        return JSONResponse(status_code=400, content={"message": "audio_path is required"})
    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        return JSONResponse(status_code=400, content={"message": f"audio_path not found: {audio_path}"})
    req["audio_path"] = audio_path
    if language not in language_code_list:
        return JSONResponse(status_code=400, content={"message": f"language not supported: {language}"})
    if engine not in ["auto", "fasterwhisper", "funasr"]:
        return JSONResponse(status_code=400, content={"message": f"engine not supported: {engine}"})
    if model_size not in get_models():
        return JSONResponse(status_code=400, content={"message": f"model_size not supported: {model_size}"})
    if precision not in ["float16", "float32", "int8"]:
        return JSONResponse(status_code=400, content={"message": f"precision not supported: {precision}"})
    return None


def get_whisper_model(model_size: str, precision: str):
    key = f"{model_size}:{precision}"
    if key in whisper_models:
        return whisper_models[key]

    if model_size not in whisper_model_paths:
        whisper_model_paths[model_size] = download_model(model_size)
    model_path = whisper_model_paths[model_size]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_path, device=device, compute_type=precision)
    whisper_models[key] = model
    return model


def asr_with_fasterwhisper(audio_path: str, language: str, model_size: str, precision: str):
    model = get_whisper_model(model_size=model_size, precision=precision)
    fw_language = None if language == "auto" else language
    segments, info = model.transcribe(
        audio=audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=700),
        language=fw_language,
    )

    # 中文/粤语默认转到 FunASR，文本稳定性通常更好
    if info.language in ["zh", "yue"]:
        text = only_asr(audio_path, language=info.language.lower())
        if text:
            return text

    text = "".join(segment.text for segment in segments).strip()
    return text


def asr_with_funasr(audio_path: str, language: str):
    lang = language if language in ["zh", "yue"] else "zh"
    return (only_asr(audio_path, language=lang) or "").strip()


async def asr_handle(req: dict):
    check_res = check_params(req)
    if check_res is not None:
        return check_res

    audio_path = req["audio_path"]
    language = req.get("language", args.language)
    engine = req.get("engine", "auto")
    model_size = req.get("model_size", args.model_size)
    precision = req.get("precision", args.precision)

    try:
        if engine == "funasr":
            text = asr_with_funasr(audio_path=audio_path, language=language)
        elif engine == "fasterwhisper":
            text = asr_with_fasterwhisper(
                audio_path=audio_path,
                language=language,
                model_size=model_size,
                precision=precision,
            )
        else:
            # auto: 中文/粤语优先 FunASR，其它语种走 Faster-Whisper
            if language in ["zh", "yue"]:
                text = asr_with_funasr(audio_path=audio_path, language=language)
                if not text:
                    text = asr_with_fasterwhisper(
                        audio_path=audio_path,
                        language=language,
                        model_size=model_size,
                        precision=precision,
                    )
            else:
                text = asr_with_fasterwhisper(
                    audio_path=audio_path,
                    language=language,
                    model_size=model_size,
                    precision=precision,
                )

        return JSONResponse(status_code=200, content={"text": text})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "asr failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/asr")
async def asr_get_endpoint(
    audio_path: str = None,
    language: str = args.language,
    engine: str = "auto",
    model_size: str = args.model_size,
    precision: str = args.precision,
):
    req = {
        "audio_path": audio_path,
        "language": language.lower() if language else "auto",
        "engine": engine,
        "model_size": model_size,
        "precision": precision,
    }
    return await asr_handle(req)


@APP.post("/asr")
async def asr_post_endpoint(request: ASRRequest):
    req = request.dict()
    req["language"] = (req.get("language") or args.language).lower()
    req["model_size"] = req.get("model_size") or args.model_size
    req["precision"] = req.get("precision") or args.precision
    return await asr_handle(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(APP, host=host, port=port, workers=1)
