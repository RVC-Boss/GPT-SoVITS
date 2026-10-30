"""GPT-SoVITS HTTP inference service.

Start: python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
Usage and API reference: README.md and docs/api.md.
"""

from tools.portable_runtime import FFMPEG_EXE, activate as activate_portable_runtime

activate_portable_runtime(change_cwd=True)

from typing import Annotated, Union, Optional

import argparse
import asyncio
import anyio
import subprocess
import wave
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
import uvicorn
from io import BytesIO
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.streaming import validate_streaming_options
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
import threading
from contextlib import asynccontextmanager

cut_method_names = get_cut_method_names()

DEFAULT_CONFIG_PATH = "GPT_SoVITS/configs/tts_infer.yaml"
tts_config = TTS_Config(DEFAULT_CONFIG_PATH)
tts_pipeline = None


@asynccontextmanager
async def lifespan(_app):
    global tts_pipeline
    if tts_pipeline is None:
        print(tts_config)
        tts_pipeline = await run_in_threadpool(TTS, tts_config)
    try:
        yield
    finally:
        for name in ("streaming_accel", "t2s_accel"):
            accelerator = getattr(tts_pipeline, name, None)
            if accelerator is not None:
                await run_in_threadpool(accelerator.close)
        tts_pipeline = None


APP = FastAPI(
    title="GPT-SoVITS 推理 API",
    description="使用已加载的 GPT/SoVITS 模型合成语音。模型和参考音频路径均由服务端读取；热切换仅作用于当前进程。",
    lifespan=lifespan,
)


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
    """Validate a request and serialize inference and stream cleanup for the shared pipeline."""

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


@APP.get("/tts")
async def tts_get_endpoint(
    request: Request,
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: Annotated[Optional[list[str]], Query()] = None,
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
    """使用查询参数合成语音，响应体为音频字节。参考音频路径由服务端读取。"""
    req = {
        "text": text,
        "text_lang": text_lang.lower() if text_lang else text_lang,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower() if prompt_lang else prompt_lang,
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
    """使用 JSON 参数合成语音，响应体为音频字节。字段与 GET /tts 对应。"""
    req = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    request_body = await raw_request.json()
    raw_streaming_mode = request_body.get("streaming_mode", request.streaming_mode)
    return await tts_handle(req, raw_streaming_mode=raw_streaming_mode)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    """预加载服务端参考音频。后续 /tts 仍须提供 ref_audio_path 和对应参考文本。"""
    try:
        async with _tts_pipeline_lock:
            await _run_in_threadpool_shielded(tts_pipeline.set_ref_audio, refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    """加载匹配当前 SoVITS 版本的 GPT 权重；只影响当前进程。"""
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
    """加载 SoVITS 权重及对应版本。切换整套声音时，随后加载匹配 GPT 和参考音频。"""
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        async with _tts_pipeline_lock:
            await _run_in_threadpool_shielded(tts_pipeline.init_vits_weights, weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS inference API")
    parser.add_argument("-c", "--tts_config", default=DEFAULT_CONFIG_PATH, help="TTS inference configuration")
    parser.add_argument("-a", "--bind_addr", default="127.0.0.1", help="default: 127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=9880, help="default: 9880")
    args = parser.parse_args()
    tts_config = TTS_Config(args.tts_config or DEFAULT_CONFIG_PATH)
    host, port = args.bind_addr, args.port
    if host == "None":
        host = None
    uvicorn.run(app=APP, host=host, port=port, workers=1)
