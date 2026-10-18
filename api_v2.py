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
    "top_k": 15,                  # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
    "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
    "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
    "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
    "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
    "gpt_streaming": false,       # bool. opt in to semantic-token streaming.
    "streaming_chunk_seconds": 2.0,
    "streaming_window_mode": "fixed",
    "streaming_lookahead_seconds": 0.12,
    "streaming_overlap_seconds": 0.04,
    "gpt_streaming_lookahead_seconds": 0.5,
    "use_flash_attention": true,
}
```

When `gpt_streaming` is enabled, use model `v2ProPlus`, `v5turbo` or `v5dev`,
integer `streaming_mode` 2 or 3, `batch_size=1`, and `media_type="raw"` or `"wav"`.
All existing text split methods (`cut0` through `cut5`) are supported: each text
segment streams through its own GPT session within one HTTP response.
`streaming_window_mode` defaults to `fixed`; `hybrid` is supported for these models.
On v2ProPlus, `min_chunk_length` sets the initial semantic block size; hybrid
windows can grow later blocks up to four times that token count. On V5,
`streaming_chunk_seconds` sets every fixed window, or the first hybrid window
of each text segment.

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

from tools.portable_runtime import FFMPEG_EXE, activate as activate_portable_runtime

activate_portable_runtime(change_cwd=True)

import os
import sys
import traceback
from typing import Generator, Union, Optional

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import asyncio
import anyio
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.streaming import validate_streaming_options
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
import threading

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

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
    top_k: int = 15
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
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    use_cuda_graph: bool = True
    repetition_penalty: float = 1.35
    sample_steps: Optional[int] = None
    cfg_rate: Optional[float] = None
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    gpt_streaming: bool = False
    streaming_chunk_seconds: float = 2.0
    streaming_window_mode: str = "fixed"
    streaming_lookahead_seconds: float = 0.12
    streaming_overlap_seconds: float = 0.04
    gpt_streaming_lookahead_seconds: float = 0.5
    use_flash_attention: bool = True


_tts_pipeline_lock = asyncio.Lock()
_STREAM_END = object()


async def _await_shielded(awaitable):
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # AnyIO cancellation is level-triggered. Shield the in-flight worker call
        # until it finishes, then let the original cancellation unwind the stream.
        with anyio.CancelScope(shield=True):
            try:
                await task
            except BaseException:
                pass
        raise


async def _run_in_threadpool_shielded(function, *args):
    return await _await_shielded(run_in_threadpool(function, *args))


def _next_or_end(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_END


async def _next_tts_chunk(iterator, skip_empty=False):
    while True:
        item = await _run_in_threadpool_shielded(_next_or_end, iterator)
        if item is _STREAM_END or not skip_empty or item[1].size > 0:
            return item


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    # Author: AkagawaTsurunaki
    # Issue:
    #   Stack overflow probabilistically occurs
    #   when the function `sf_writef_short` of `libsndfile_64bit.dll` is called
    #   using the Python library `soundfile`
    # Note:
    #   This is an issue related to `libsndfile`, not this project itself.
    #   It happens when you generate a large audio tensor (about 499804 frames in my PC)
    #   and try to convert it to an ogg file.
    # Related:
    #   https://github.com/RVC-Boss/GPT-SoVITS/issues/1199
    #   https://github.com/libsndfile/libsndfile/issues/1023
    #   https://github.com/bastibe/python-soundfile/issues/396
    # Suggestion:
    #   Or split the whole audio data into smaller audio segment to avoid stack overflow?

    def handle_pack_ogg():
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)



    # See: https://docs.python.org/3/library/threading.html
    # The stack size of this thread is at least 32768
    # If stack overflow error still occurs, just modify the `stack_size`.
    # stack_size = n * 4096, where n should be a positive integer.
    # Here we chose n = 4096.
    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as e:
        # If changing the thread stack size is unsupported, a RuntimeError is raised.
        print("RuntimeError: {}".format(e))
        print("Changing the thread stack size is unsupported.")
    except ValueError as e:
        # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
        print("ValueError: {}".format(e))
        print("The specified stack size is invalid.")

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
            str(FFMPEG_EXE),
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
        os.execl(sys.executable, sys.executable, "-I", *argv)
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
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    # elif media_type == "ogg" and not streaming_mode:
    #     return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


async def tts_handle(req: dict, raw_streaming_mode=None):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 15,                  # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "parallel_infer": True,       # bool. whether to use parallel inference.
                "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
                "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
                "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
                "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
                "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
            }
    returns:
        StreamingResponse: audio stream response.
    """

    await _tts_pipeline_lock.acquire()
    tts_generator = None
    response_owns_lock = False
    stream_cleanup_task = None

    async def cleanup_stream():
        nonlocal stream_cleanup_task
        if stream_cleanup_task is None:
            async def close_and_release():
                try:
                    if tts_generator is not None:
                        close = getattr(tts_generator, "close", None)
                        if callable(close):
                            await _run_in_threadpool_shielded(close)
                finally:
                    _tts_pipeline_lock.release()

            stream_cleanup_task = asyncio.create_task(close_and_release())
        await _await_shielded(stream_cleanup_task)

    try:
        check_res = check_params(req)
        if check_res is not None:
            return check_res

        gpt_streaming = req.get("gpt_streaming", False)
        if gpt_streaming:
            try:
                validation_req = dict(req)
                if raw_streaming_mode is not None:
                    validation_req["streaming_mode"] = raw_streaming_mode
                validate_streaming_options(
                    validation_req,
                    getattr(tts_pipeline.configs, "version", tts_config.version),
                    api_mode=True,
                )
            except (TypeError, ValueError) as e:
                return JSONResponse(status_code=400, content={"message": str(e)})

        streaming_mode = req.get("streaming_mode", False)
        return_fragment = req.get("return_fragment", False)
        media_type = req.get("media_type", "wav")

        if streaming_mode == 0:
            streaming_mode = False
            return_fragment = False
            fixed_length_chunk = False
        elif streaming_mode == 1:
            streaming_mode = False
            return_fragment = True
            fixed_length_chunk = False
        elif streaming_mode == 2:
            streaming_mode = True
            return_fragment = False
            fixed_length_chunk = False
        elif streaming_mode == 3:
            streaming_mode = True
            return_fragment = False
            fixed_length_chunk = True
        else:
            return JSONResponse(status_code=400, content={"message": f"the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)"})

        req["streaming_mode"] = streaming_mode
        req["return_fragment"] = return_fragment
        req["fixed_length_chunk"] = fixed_length_chunk

        print(f"{streaming_mode} {return_fragment} {fixed_length_chunk}")

        streaming_mode = streaming_mode or return_fragment
        tts_generator = await _run_in_threadpool_shielded(tts_pipeline.run, req)

        if streaming_mode:
            first_chunk = _STREAM_END
            if gpt_streaming:
                first_chunk = await _next_tts_chunk(tts_generator, skip_empty=True)
                if first_chunk is _STREAM_END:
                    return JSONResponse(
                        status_code=400,
                        content={"message": "tts failed", "Exception": "stream produced no PCM audio"},
                    )

            async def streaming_generator():
                nonlocal media_type
                first_audio = True
                item = first_chunk
                try:
                    while True:
                        if item is _STREAM_END:
                            item = await _next_tts_chunk(tts_generator, skip_empty=gpt_streaming)
                        if item is _STREAM_END:
                            break

                        sr, chunk = item
                        if first_audio and media_type == "wav":
                            yield wave_header_chunk(sample_rate=sr)
                            media_type = "raw"
                        first_audio = False
                        packed = await _run_in_threadpool_shielded(
                            pack_audio, BytesIO(), chunk, sr, media_type
                        )
                        yield packed.getvalue()
                        item = _STREAM_END
                finally:
                    await cleanup_stream()

            response = StreamingResponse(
                streaming_generator(),
                media_type=f"audio/{media_type}",
                background=BackgroundTask(cleanup_stream),
            )
            response_owns_lock = True
            return response

        item = await _next_tts_chunk(tts_generator)
        if item is _STREAM_END:
            raise RuntimeError("TTS produced no audio")
        sr, audio_data = item
        packed = await _run_in_threadpool_shielded(pack_audio, BytesIO(), audio_data, sr, media_type)
        return Response(packed.getvalue(), media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})
    finally:
        if not response_owns_lock:
            try:
                if tts_generator is not None:
                    close = getattr(tts_generator, "close", None)
                    if callable(close):
                        await _run_in_threadpool_shielded(close)
            finally:
                _tts_pipeline_lock.release()


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    request: Request,
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    parallel_infer: bool = True,
    use_cuda_graph: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: Optional[int] = None,
    cfg_rate: Optional[float] = None,
    super_sampling: bool = False,
    streaming_mode: Union[bool, int] = False,
    overlap_length: int = 2,
    min_chunk_length: int = 16,
    gpt_streaming: bool = False,
    streaming_chunk_seconds: float = 2.0,
    streaming_window_mode: str = "fixed",
    streaming_lookahead_seconds: float = 0.12,
    streaming_overlap_seconds: float = 0.04,
    gpt_streaming_lookahead_seconds: float = 0.5,
    use_flash_attention: bool = True,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
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
        "use_cuda_graph": use_cuda_graph,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": None if sample_steps is None else int(sample_steps),
        "cfg_rate": None if cfg_rate is None else float(cfg_rate),
        "super_sampling": super_sampling,
        "overlap_length": int(overlap_length),
        "min_chunk_length": int(min_chunk_length),
        "gpt_streaming": gpt_streaming,
        "streaming_chunk_seconds": float(streaming_chunk_seconds),
        "streaming_window_mode": streaming_window_mode,
        "streaming_lookahead_seconds": float(streaming_lookahead_seconds),
        "streaming_overlap_seconds": float(streaming_overlap_seconds),
        "gpt_streaming_lookahead_seconds": float(gpt_streaming_lookahead_seconds),
        "use_flash_attention": use_flash_attention,
    }
    raw_streaming_mode = request.query_params.get("streaming_mode")
    if raw_streaming_mode in {"2", "3"}:
        raw_streaming_mode = int(raw_streaming_mode)
    return await tts_handle(req, raw_streaming_mode=raw_streaming_mode)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request, raw_request: Request):
    req = request.dict()
    request_body = await raw_request.json()
    raw_streaming_mode = request_body.get("streaming_mode", request.streaming_mode)
    return await tts_handle(req, raw_streaming_mode=raw_streaming_mode)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        async with _tts_pipeline_lock:
            await _run_in_threadpool_shielded(tts_pipeline.set_ref_audio, refer_audio_path)
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
        async with _tts_pipeline_lock:
            await _run_in_threadpool_shielded(tts_pipeline.init_t2s_weights, weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        async with _tts_pipeline_lock:
            await _run_in_threadpool_shielded(tts_pipeline.init_vits_weights, weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
