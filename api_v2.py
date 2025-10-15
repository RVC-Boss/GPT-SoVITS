"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                   # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "streaming_mode": False,      # bool. whether to return a streaming response.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
    "super_sampling": False       # bool. whether to use super-sampling for audio when using VITS model V3.
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

"""

import os
import sys
import traceback
from typing import Generator, Dict, Any, Optional
import uuid
import asyncio
from datetime import datetime

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
import json
import yaml

# Import config variables (avoiding webui to prevent Gradio loading)
from config import (
    exp_root,
    python_exec,
    is_half,
    GPU_INDEX,
    infer_device,
    SoVITS_weight_version2root,
    GPT_weight_version2root,
)

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

# GPU helper functions (replicated from webui.py to avoid import)
set_gpu_numbers = GPU_INDEX
default_gpu_numbers = infer_device.index if hasattr(infer_device, 'index') else 0

def fix_gpu_number(input_val):
    """Fix GPU number to be within valid range."""
    try:
        if int(input_val) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input_val
    return input_val

def fix_gpu_numbers(inputs):
    """Fix multiple GPU numbers separated by comma."""
    output = []
    try:
        for input_val in inputs.split(","):
            output.append(str(fix_gpu_number(input_val)))
        return ",".join(output)
    except:
        return inputs

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


class SpeechSlicingRequest(BaseModel):
    inp: str
    opt_root: str
    threshold: str = "-34"
    min_length: str = "4000"
    min_interval: str = "300"
    hop_size: str = "10"
    max_sil_kept: str = "500"
    _max: float = 0.9
    alpha: float = 0.25
    n_parts: int = 4


class STTRequest(BaseModel):
    input_folder: str
    output_folder: str
    model_path: str = "tools/asr/models/faster-whisper-large-v3"
    language: str = "auto"
    precision: str = "float32"


class DatasetFormattingRequest(BaseModel):
    inp_text: str
    inp_wav_dir: str
    exp_name: str
    version: str = "v4"
    gpu_numbers: str = "0-0"
    bert_pretrained_dir: str = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    ssl_pretrained_dir: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    pretrained_s2G_path: str = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"


class FineTuneSoVITSRequest(BaseModel):
    version: str = "v4"
    batch_size: int = 2
    total_epoch: int = 2
    exp_name: str
    text_low_lr_rate: float = 0.4
    if_save_latest: bool = True
    if_save_every_weights: bool = True
    save_every_epoch: int = 1
    gpu_numbers1Ba: str = "0"
    pretrained_s2G: str = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
    pretrained_s2D: str = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Dv4.pth"
    if_grad_ckpt: bool = False
    lora_rank: str = "32"


class FineTuneGPTRequest(BaseModel):
    batch_size: int = 8
    total_epoch: int = 15
    exp_name: str
    if_dpo: bool = False
    if_save_latest: bool = True
    if_save_every_weights: bool = True
    save_every_epoch: int = 5
    gpu_numbers: str = "0"
    pretrained_s1: str = "GPT_SoVITS/pretrained_models/s1v3.ckpt"


jobs: Dict[str, Dict[str, Any]] = {}


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
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


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
                "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                "super_sampling": False,       # bool. whether to use super-sampling for audio when using VITS model V3.
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    try:
        tts_generator = tts_pipeline.run(req)

        if streaming_mode:

            def streaming_generator(tts_generator: Generator, media_type: str):
                if_frist_chunk = True
                for sr, chunk in tts_generator:
                    if if_frist_chunk and media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        media_type = "raw"
                        if_frist_chunk = False
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(
                streaming_generator(
                    tts_generator,
                    media_type,
                ),
                media_type=f"audio/{media_type}",
            )

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut0",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
):
    req = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    # DEBUG: Print received payload
    print(f"\n{'='*80}")
    print(f"[TTS DEBUG] Received request:")
    print(f"  text: {request.text[:100] if len(request.text) > 100 else request.text}")  # Truncate long text
    print(f"  text_lang: {request.text_lang}")
    print(f"  ref_audio_path: {request.ref_audio_path}")
    print(f"  prompt_text: {request.prompt_text[:100] if request.prompt_text and len(request.prompt_text) > 100 else request.prompt_text}")
    print(f"  prompt_lang: {request.prompt_lang}")
    print(f"  top_k: {request.top_k}")
    print(f"  top_p: {request.top_p}")
    print(f"  temperature: {request.temperature}")
    print(f"  text_split_method: {request.text_split_method}")
    print(f"  batch_size: {request.batch_size}")
    print(f"  speed_factor: {request.speed_factor}")
    print(f"  streaming_mode: {request.streaming_mode}")
    print(f"{'='*80}\n")

    req = request.dict()
    return await tts_handle(req)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


async def execute_job_async(job_id: str, operation_func, *args, **kwargs):
    """
    Execute a job asynchronously in background.

    Args:
        job_id: Unique job identifier
        operation_func: Function to execute (from webui.py)
        args, kwargs: Arguments for the operation function
    """
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()

    try:
        result = await asyncio.to_thread(operation_func, *args, **kwargs)

        if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
            final_result = None
            for item in result:
                final_result = item
            result = final_result

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()
        jobs[job_id]["failed_at"] = datetime.now().isoformat()


@APP.get("/jobs/{job_id}")
@APP.get("/job-status/{job_id}")  # Alias for compatibility
async def get_job_status(job_id: str):
    """
    Get job status and result.

    Returns:
        {
            "job_id": str,
            "status": "queued" | "running" | "completed" | "failed",
            "result": Any (if completed),
            "error": str (if failed),
            "created_at": str,
            "started_at": str (if running/completed/failed),
            "completed_at": str (if completed),
            "failed_at": str (if failed)
        }
    """
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"message": "job not found"})

    job_data = jobs[job_id].copy()
    job_data["job_id"] = job_id
    return JSONResponse(status_code=200, content=job_data)


async def execute_speech_slicing_direct(job_id: str, request: SpeechSlicingRequest):
    """
    Execute speech slicing by directly calling slice_audio.py subprocess.
    Replaces webui.open_slice() to avoid Gradio dependency.
    """
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()

    try:
        # Prepare environment with PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([now_dir, os.path.join(now_dir, "GPT_SoVITS")])

        # Create processes for parallel slicing (n_parts)
        processes = []
        for i_part in range(request.n_parts):
            cmd = [
                python_exec,
                "tools/slice_audio.py",
                request.inp,
                request.opt_root,
                str(request.threshold),
                str(request.min_length),
                str(request.min_interval),
                str(request.hop_size),
                str(request.max_sil_kept),
                str(request._max),
                str(request.alpha),
                str(i_part),
                str(request.n_parts),
            ]
            print(f"[SPEECH SLICING] Executing: {' '.join(cmd)}")
            p = subprocess.Popen(cmd, env=env, cwd=now_dir)
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.wait()

        # Check if any process failed
        exit_codes = [p.returncode for p in processes]
        if any(code != 0 for code in exit_codes):
            raise Exception(f"Speech slicing failed with exit codes: {exit_codes}")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "output_dir": request.opt_root,
            "file_count": request.n_parts
        }
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()
        jobs[job_id]["failed_at"] = datetime.now().isoformat()


@APP.post("/preprocessing/speech-slicing")
async def speech_slicing_endpoint(request: SpeechSlicingRequest):
    """
    Start speech slicing job.

    Directly executes tools/slice_audio.py (no webui dependency).
    """
    # DEBUG: Print received payload
    print(f"\n{'='*80}")
    print(f"[SPEECH SLICING DEBUG] Received request:")
    print(f"  inp: {request.inp}")
    print(f"  opt_root: {request.opt_root}")
    print(f"  threshold: {request.threshold}")
    print(f"  min_length: {request.min_length}")
    print(f"  min_interval: {request.min_interval}")
    print(f"  hop_size: {request.hop_size}")
    print(f"  max_sil_kept: {request.max_sil_kept}")
    print(f"  _max: {request._max}")
    print(f"  alpha: {request.alpha}")
    print(f"  n_parts: {request.n_parts}")
    print(f"{'='*80}\n")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "operation": "speech_slicing",
        "created_at": datetime.now().isoformat()
    }

    try:
        asyncio.create_task(execute_speech_slicing_direct(job_id, request))
        return JSONResponse(status_code=200, content={"job_id": job_id, "status": "queued"})
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        return JSONResponse(status_code=500, content={"message": "failed to start job", "error": str(e)})


@APP.post("/preprocessing/stt")
async def stt_endpoint(request: STTRequest):
    """
    Start STT (Speech-to-Text) job.

    Wraps tools/asr/fasterwhisper_asr.execute_asr()
    """
    # DEBUG: Print received payload
    print(f"\n{'='*80}")
    print(f"[STT DEBUG] Received STT request:")
    print(request)
    print(f"{'='*80}\n")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "operation": "stt",
        "created_at": datetime.now().isoformat()
    }

    try:
        from tools.asr.fasterwhisper_asr import execute_asr

        asyncio.create_task(execute_job_async(
            job_id,
            execute_asr,
            request.input_folder,
            request.output_folder,
            request.model_path,
            request.language,
            request.precision
        ))

        return JSONResponse(status_code=200, content={"job_id": job_id, "status": "queued"})
    except Exception as e:
        print(f"[STT ERROR] Failed to start STT job: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        return JSONResponse(status_code=500, content={"message": "failed to start job", "error": str(e)})


async def execute_dataset_formatting(job_id: str, request: DatasetFormattingRequest):
    """
    Execute dataset formatting sequentially: open1a -> open1b -> open1c
    Directly executes subprocess (no webui dependency).
    """
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()
    jobs[job_id]["current_stage"] = "open1a"

    try:
        opt_dir = f"{exp_root}/{request.exp_name}"
        os.makedirs(opt_dir, exist_ok=True)

        # Parse GPU numbers
        gpu_names = request.gpu_numbers.split("-")
        all_parts = len(gpu_names)

        # Stage 1a: Get text features
        print(f"[DATASET FORMATTING] Starting open1a...")
        for i_part in range(all_parts):
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": os.pathsep.join([now_dir, os.path.join(now_dir, "GPT_SoVITS")]),
                "inp_text": request.inp_text,
                "inp_wav_dir": request.inp_wav_dir,
                "exp_name": request.exp_name,
                "opt_dir": opt_dir,
                "bert_pretrained_dir": request.bert_pretrained_dir,
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                "is_half": str(is_half),
            })
            cmd = [python_exec, "GPT_SoVITS/prepare_datasets/1-get-text.py"]
            print(f"[DATASET FORMATTING] Executing 1a part {i_part}: {' '.join(cmd)}")
            await asyncio.to_thread(subprocess.run, cmd, env=env, cwd=now_dir, check=True)

        # Merge text files from 1a stage
        opt = []
        path_text = f"{opt_dir}/2-name2text.txt"
        for i_part in range(all_parts):
            text_path = f"{opt_dir}/2-name2text-{i_part}.txt"
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf8") as f:
                    opt += f.read().strip("\n").split("\n")
                os.remove(text_path)

        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")

        # Stage 1b: Get hubert features
        jobs[job_id]["current_stage"] = "open1b"
        print(f"[DATASET FORMATTING] Starting open1b...")
        sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"

        for i_part in range(all_parts):
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": os.pathsep.join([now_dir, os.path.join(now_dir, "GPT_SoVITS")]),
                "inp_text": request.inp_text,
                "inp_wav_dir": request.inp_wav_dir,
                "exp_name": request.exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": request.ssl_pretrained_dir,
                "sv_path": sv_path,
                "is_half": str(is_half),
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
            })
            cmd = [python_exec, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"]
            print(f"[DATASET FORMATTING] Executing 1b part {i_part}: {' '.join(cmd)}")
            await asyncio.to_thread(subprocess.run, cmd, env=env, cwd=now_dir, check=True)

        # For v2Pro version, also run 2-get-sv.py
        if "Pro" in request.version:
            for i_part in range(all_parts):
                env = os.environ.copy()
                env.update({
                    "PYTHONPATH": os.pathsep.join([now_dir, os.path.join(now_dir, "GPT_SoVITS")]),
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                    "exp_dir": opt_dir,
                    "sv_path": sv_path,
                    "is_half": str(is_half),
                })
                cmd = [python_exec, "GPT_SoVITS/prepare_datasets/2-get-sv.py"]
                print(f"[DATASET FORMATTING] Executing 2-get-sv part {i_part}: {' '.join(cmd)}")
                await asyncio.to_thread(subprocess.run, cmd, env=env, cwd=now_dir, check=True)

        # Stage 1c: Get semantic features
        jobs[job_id]["current_stage"] = "open1c"
        print(f"[DATASET FORMATTING] Starting open1c...")
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if request.version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{request.version}.json"
        )

        for i_part in range(all_parts):
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": os.pathsep.join([now_dir, os.path.join(now_dir, "GPT_SoVITS")]),
                "inp_text": request.inp_text,
                "exp_name": request.exp_name,
                "opt_dir": opt_dir,
                "pretrained_s2G": request.pretrained_s2G_path,
                "s2config_path": config_file,
                "is_half": str(is_half),
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
            })
            cmd = [python_exec, "GPT_SoVITS/prepare_datasets/3-get-semantic.py"]
            print(f"[DATASET FORMATTING] Executing 1c part {i_part}: {' '.join(cmd)}")
            await asyncio.to_thread(subprocess.run, cmd, env=env, cwd=now_dir, check=True)

        # Merge semantic files (from open1c logic in webui.py)
        opt = ["item_name\tsemantic_audio"]
        path_semantic = f"{opt_dir}/6-name2semantic.tsv"
        for i_part in range(all_parts):
            semantic_path = f"{opt_dir}/6-name2semantic-{i_part}.tsv"
            if os.path.exists(semantic_path):
                with open(semantic_path, "r", encoding="utf8") as f:
                    opt += f.read().strip("\n").split("\n")
                os.remove(semantic_path)

        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt))

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "exp_name": request.exp_name,
            "stages_completed": ["open1a", "open1b", "open1c"]
        }
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()
        jobs[job_id]["failed_at"] = datetime.now().isoformat()


@APP.post("/training/format-dataset")
async def format_dataset_endpoint(request: DatasetFormattingRequest):
    """
    Start dataset formatting job (open1a -> open1b -> open1c).

    Wraps webui.open1a(), open1b(), open1c() sequentially.
    """
    # DEBUG: Print received payload
    print(f"\n{'='*80}")
    print(f"[DATASET FORMATTING DEBUG] Received request:")
    print(f"  version: {request.version}")
    print(f"  inp_text: {request.inp_text}")
    print(f"  inp_wav_dir: {request.inp_wav_dir}")
    print(f"  exp_name: {request.exp_name}")
    print(f"  gpu_numbers1a: {request.gpu_numbers}")
    print(f"  bert_pretrained_dir: {request.bert_pretrained_dir}")
    print(f"  ssl_pretrained_dir: {request.ssl_pretrained_dir}")
    print(f"  pretrained_s2G_path: {request.pretrained_s2G_path}")
    print(f"{'='*80}\n")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "operation": "format_dataset",
        "created_at": datetime.now().isoformat()
    }

    try:
        asyncio.create_task(execute_dataset_formatting(job_id, request))
        return JSONResponse(status_code=200, content={"job_id": job_id, "status": "queued"})
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        return JSONResponse(status_code=500, content={"message": "failed to start job", "error": str(e)})


async def execute_fine_tune_sovits_direct(job_id: str, request: FineTuneSoVITSRequest):
    """
    Execute SoVITS fine-tuning by directly calling s2_train.py subprocess.
    Replaces webui.open1Ba() to avoid Gradio dependency.
    """
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()

    try:
        s2_dir = f"{exp_root}/{request.exp_name}"
        os.makedirs(f"{s2_dir}/logs_s2_{request.version}", exist_ok=True)

        # Load config template
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if request.version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{request.version}.json"
        )
        with open(config_file) as f:
            data = json.loads(f.read())

        # Update config with request parameters
        batch_size = request.batch_size
        if is_half == False:
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)

        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = request.total_epoch
        data["train"]["text_low_lr_rate"] = request.text_low_lr_rate
        data["train"]["pretrained_s2G"] = request.pretrained_s2G
        data["train"]["pretrained_s2D"] = request.pretrained_s2D
        data["train"]["if_save_latest"] = request.if_save_latest
        data["train"]["if_save_every_weights"] = request.if_save_every_weights
        data["train"]["save_every_epoch"] = request.save_every_epoch
        data["train"]["gpu_numbers"] = request.gpu_numbers1Ba
        data["train"]["grad_ckpt"] = request.if_grad_ckpt
        data["train"]["lora_rank"] = request.lora_rank
        data["model"]["version"] = request.version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_version2root[request.version]
        data["name"] = request.exp_name
        data["version"] = request.version

        # Write temporary config
        tmp_config_path = f"{now_dir}/TEMP/tmp_s2.json"
        os.makedirs(f"{now_dir}/TEMP", exist_ok=True)
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))

        # Prepare environment with PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([now_dir, os.path.join(now_dir, "GPT_SoVITS")])

        # Determine training script based on version
        if request.version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
            cmd = [python_exec, "GPT_SoVITS/s2_train.py", "--config", tmp_config_path]
        else:
            cmd = [python_exec, "GPT_SoVITS/s2_train_v3_lora.py", "--config", tmp_config_path]

        print(f"[SOVITS FINE-TUNING] Executing: {' '.join(cmd)}")
        result = await asyncio.to_thread(subprocess.run, cmd, env=env, cwd=now_dir, check=True)

        # Find latest SoVITS checkpoint
        sovits_weights_dir = data["save_weight_dir"]
        latest_sovits_checkpoint = None

        if os.path.exists(sovits_weights_dir):
            import re
            pattern = re.compile(rf"^{re.escape(request.exp_name)}_e(\d+)_s(\d+)_l(\d+)\.pth$")
            checkpoints = []
            for filename in os.listdir(sovits_weights_dir):
                match = pattern.match(filename)
                if match:
                    epoch = int(match.group(1))
                    step = int(match.group(2))
                    checkpoints.append((epoch, step, filename))

            if checkpoints:
                checkpoints.sort(reverse=True)
                latest_filename = checkpoints[0][2]
                latest_sovits_checkpoint = os.path.join(sovits_weights_dir, latest_filename)
                print(f"[SOVITS FINE-TUNING] Latest checkpoint: {latest_sovits_checkpoint}")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "exp_name": request.exp_name,
            "config_path": tmp_config_path,
            "checkpoint_path": latest_sovits_checkpoint,
            "sovits_checkpoint_path": latest_sovits_checkpoint
        }
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()
        jobs[job_id]["failed_at"] = datetime.now().isoformat()


@APP.post("/training/fine-tune-sovits")
async def fine_tune_sovits_endpoint(request: FineTuneSoVITSRequest):
    """
    Start SoVITS fine-tuning job.

    Directly executes s2_train.py (no webui dependency).
    """
    # DEBUG: Print received payload
    print(f"\n{'='*80}")
    print(f"[SOVITS FINE-TUNING DEBUG] Received request:")
    print(f"  version: {request.version}")
    print(f"  batch_size: {request.batch_size}")
    print(f"  total_epoch: {request.total_epoch}")
    print(f"  exp_name: {request.exp_name}")
    print(f"  text_low_lr_rate: {request.text_low_lr_rate}")
    print(f"  if_save_latest: {request.if_save_latest}")
    print(f"  if_save_every_weights: {request.if_save_every_weights}")
    print(f"  save_every_epoch: {request.save_every_epoch}")
    print(f"  gpu_numbers1Ba: {request.gpu_numbers1Ba}")
    print(f"  pretrained_s2G: {request.pretrained_s2G}")
    print(f"  pretrained_s2D: {request.pretrained_s2D}")
    print(f"  if_grad_ckpt: {request.if_grad_ckpt}")
    print(f"  lora_rank: {request.lora_rank}")
    print(f"{'='*80}\n")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "operation": "fine_tune_sovits",
        "created_at": datetime.now().isoformat()
    }

    try:
        asyncio.create_task(execute_fine_tune_sovits_direct(job_id, request))
        return JSONResponse(status_code=200, content={"job_id": job_id, "status": "queued"})
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        return JSONResponse(status_code=500, content={"message": "failed to start job", "error": str(e)})


@APP.post("/training/fine-tune-gpt")
async def fine_tune_gpt_endpoint(request: FineTuneGPTRequest):
    """
    Start GPT fine-tuning job.

    Wraps webui.open1Bb()
    """
    # DEBUG: Print received payload
    print(f"\n{'='*80}")
    print(f"[GPT FINE-TUNING DEBUG] Received request:")
    print(f"  batch_size: {request.batch_size}")
    print(f"  total_epoch: {request.total_epoch}")
    print(f"  exp_name: {request.exp_name}")
    print(f"  if_dpo: {request.if_dpo}")
    print(f"  if_save_latest: {request.if_save_latest}")
    print(f"  if_save_every_weights: {request.if_save_every_weights}")
    print(f"  save_every_epoch: {request.save_every_epoch}")
    print(f"  gpu_numbers: {request.gpu_numbers}")
    print(f"  pretrained_s1: {request.pretrained_s1}")
    print(f"{'='*80}\n")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "operation": "fine_tune_gpt",
        "created_at": datetime.now().isoformat()
    }

    try:
        asyncio.create_task(execute_fine_tune_gpt_direct(job_id, request))
        return JSONResponse(status_code=200, content={"job_id": job_id, "status": "queued"})
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        return JSONResponse(status_code=500, content={"message": "failed to start job", "error": str(e)})


async def execute_fine_tune_gpt_direct(job_id: str, request: FineTuneGPTRequest):
    """
    Execute GPT fine-tuning by directly calling s1_train.py subprocess.
    Replaces webui.open1Bb() to avoid Gradio dependency.
    """
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()

    try:
        s1_dir = f"{exp_root}/{request.exp_name}"
        os.makedirs(f"{s1_dir}/logs_s1", exist_ok=True)

        # Determine version (from webui.py line 606)
        version = os.environ.get("version", "v4")

        # Load config template
        config_path = (
            "GPT_SoVITS/configs/s1longer.yaml" if version == "v1"
            else "GPT_SoVITS/configs/s1longer-v2.yaml"
        )
        with open(config_path) as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Update config with request parameters
        batch_size = request.batch_size
        if is_half == False:
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)

        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = request.total_epoch
        data["pretrained_s1"] = request.pretrained_s1
        data["train"]["save_every_n_epoch"] = request.save_every_epoch
        data["train"]["if_save_every_weights"] = request.if_save_every_weights
        data["train"]["if_save_latest"] = request.if_save_latest
        data["train"]["if_dpo"] = request.if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_version2root[version]
        data["train"]["exp_name"] = request.exp_name
        data["train_semantic_path"] = f"{s1_dir}/6-name2semantic.tsv"
        data["train_phoneme_path"] = f"{s1_dir}/2-name2text.txt"
        data["output_dir"] = f"{s1_dir}/logs_s1_{version}"

        # Set environment variables for GPU and PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([now_dir, os.path.join(now_dir, "GPT_SoVITS")])
        env["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(request.gpu_numbers.replace("-", ","))
        env["hz"] = "25hz"

        # Write temporary config
        tmp_config_path = f"{now_dir}/TEMP/tmp_s1.yaml"
        os.makedirs(f"{now_dir}/TEMP", exist_ok=True)
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))

        # Execute training
        cmd = [python_exec, "GPT_SoVITS/s1_train.py", "--config_file", tmp_config_path]
        print(f"[GPT FINE-TUNING] Executing: {' '.join(cmd)}")
        result = await asyncio.to_thread(subprocess.run, cmd, env=env, cwd=now_dir, check=True)

        # Find latest GPT checkpoint
        gpt_weights_dir = data["train"]["half_weights_save_dir"]
        latest_gpt_checkpoint = None

        if os.path.exists(gpt_weights_dir):
            import re
            pattern = re.compile(rf"^{re.escape(request.exp_name)}-e(\d+)\.ckpt$")
            checkpoints = []
            for filename in os.listdir(gpt_weights_dir):
                match = pattern.match(filename)
                if match:
                    epoch = int(match.group(1))
                    checkpoints.append((epoch, filename))

            if checkpoints:
                checkpoints.sort(reverse=True)
                latest_filename = checkpoints[0][1]
                latest_gpt_checkpoint = os.path.join(gpt_weights_dir, latest_filename)
                print(f"[GPT FINE-TUNING] Latest checkpoint: {latest_gpt_checkpoint}")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "exp_name": request.exp_name,
            "config_path": tmp_config_path,
            "checkpoint_path": latest_gpt_checkpoint,
            "gpt_checkpoint_path": latest_gpt_checkpoint
        }
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()
        jobs[job_id]["failed_at"] = datetime.now().isoformat()


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
