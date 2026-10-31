"""Inference and model-control HTTP routes."""

import asyncio
from io import BytesIO
from typing import Annotated, Optional, Union

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from GPT_SoVITS.inference.segmentation import get_method_names as get_cut_method_names
from GPT_SoVITS.inference.streaming import validate_streaming_options

from . import app as service_app
from .audio_response import pack_audio, wave_header_chunk
from .concurrency import (
    STREAM_END,
    await_shielded,
    close_generator,
    next_tts_chunk,
    run_in_threadpool_shielded,
)
from .schemas import TTS_Request, get_request_dict, raw_get_streaming_mode, request_model_to_dict


cut_method_names = get_cut_method_names()


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")
    config = service_app.state.tts_config

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    if text_lang.lower() not in config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    if prompt_lang.lower() not in config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_split_method:{text_split_method} is not supported"},
        )
    return None


async def tts_handle(req: dict, raw_streaming_mode=None):
    """Validate, serialize inference, and retain the lock through streaming."""

    state = service_app.state
    await state.pipeline_lock.acquire()
    tts_generator = None
    response_owns_lock = False
    stream_cleanup_task = None

    async def cleanup_stream():
        nonlocal stream_cleanup_task
        if stream_cleanup_task is None:

            async def close_and_release():
                try:
                    await close_generator(tts_generator)
                finally:
                    state.pipeline_lock.release()

            stream_cleanup_task = asyncio.create_task(close_and_release())
        await await_shielded(stream_cleanup_task)

    try:
        check_result = check_params(req)
        if check_result is not None:
            return check_result

        gpt_streaming = req.get("gpt_streaming", False)
        pipeline = state.tts_pipeline
        if gpt_streaming:
            try:
                validation_req = dict(req)
                if raw_streaming_mode is not None:
                    validation_req["streaming_mode"] = raw_streaming_mode
                validate_streaming_options(
                    validation_req,
                    getattr(pipeline.configs, "version", state.tts_config.version),
                    api_mode=True,
                )
            except (TypeError, ValueError) as error:
                return JSONResponse(status_code=400, content={"message": str(error)})

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
            return JSONResponse(
                status_code=400,
                content={
                    "message": "the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)"
                },
            )

        req["streaming_mode"] = streaming_mode
        req["return_fragment"] = return_fragment
        req["fixed_length_chunk"] = fixed_length_chunk

        streaming_mode = streaming_mode or return_fragment
        tts_generator = await run_in_threadpool_shielded(pipeline.run, req)

        if streaming_mode:
            first_chunk = STREAM_END
            if gpt_streaming:
                first_chunk = await next_tts_chunk(tts_generator, skip_empty=True)
                if first_chunk is STREAM_END:
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
                        if item is STREAM_END:
                            item = await next_tts_chunk(tts_generator, skip_empty=gpt_streaming)
                        if item is STREAM_END:
                            break

                        sample_rate, chunk = item
                        if first_audio and media_type == "wav":
                            yield wave_header_chunk(sample_rate=sample_rate)
                            media_type = "raw"
                        first_audio = False
                        packed = await run_in_threadpool_shielded(
                            pack_audio, BytesIO(), chunk, sample_rate, media_type
                        )
                        yield packed.getvalue()
                        item = STREAM_END
                finally:
                    await cleanup_stream()

            response = StreamingResponse(
                streaming_generator(),
                media_type=f"audio/{media_type}",
                background=BackgroundTask(cleanup_stream),
            )
            response_owns_lock = True
            return response

        item = await next_tts_chunk(tts_generator)
        if item is STREAM_END:
            raise RuntimeError("TTS produced no audio")
        sample_rate, audio_data = item
        packed = await run_in_threadpool_shielded(pack_audio, BytesIO(), audio_data, sample_rate, media_type)
        return Response(packed.getvalue(), media_type=f"audio/{media_type}")
    except Exception as error:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(error)})
    finally:
        if not response_owns_lock:
            try:
                await close_generator(tts_generator)
            finally:
                state.pipeline_lock.release()


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

    req = get_request_dict(
        text=text,
        text_lang=text_lang,
        ref_audio_path=ref_audio_path,
        aux_ref_audio_paths=aux_ref_audio_paths,
        prompt_lang=prompt_lang,
        prompt_text=prompt_text,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        text_split_method=text_split_method,
        batch_size=batch_size,
        batch_threshold=batch_threshold,
        split_bucket=split_bucket,
        speed_factor=speed_factor,
        fragment_interval=fragment_interval,
        seed=seed,
        media_type=media_type,
        parallel_infer=parallel_infer,
        use_cuda_graph=use_cuda_graph,
        repetition_penalty=repetition_penalty,
        sample_steps=sample_steps,
        cfg_rate=cfg_rate,
        super_sampling=super_sampling,
        streaming_mode=streaming_mode,
        overlap_length=overlap_length,
        min_chunk_length=min_chunk_length,
        gpt_streaming=gpt_streaming,
        streaming_chunk_seconds=streaming_chunk_seconds,
        streaming_window_mode=streaming_window_mode,
        streaming_lookahead_seconds=streaming_lookahead_seconds,
        streaming_overlap_seconds=streaming_overlap_seconds,
        gpt_streaming_lookahead_seconds=gpt_streaming_lookahead_seconds,
        use_flash_attention=use_flash_attention,
    )
    return await tts_handle(req, raw_streaming_mode=raw_get_streaming_mode(request))


async def tts_post_endpoint(request: TTS_Request, raw_request: Request):
    """使用 JSON 参数合成语音，响应体为音频字节。字段与 GET /tts 对应。"""

    req = request_model_to_dict(request)
    request_body = await raw_request.json()
    raw_streaming_mode = request_body.get("streaming_mode", request.streaming_mode)
    return await tts_handle(req, raw_streaming_mode=raw_streaming_mode)


async def set_refer_aduio(refer_audio_path: str = None):
    """预加载服务端参考音频。后续 /tts 仍须提供 ref_audio_path 和对应参考文本。"""

    try:
        async with service_app.state.pipeline_lock:
            await run_in_threadpool_shielded(
                service_app.state.tts_pipeline.set_ref_audio, refer_audio_path
            )
    except Exception as error:
        return JSONResponse(
            status_code=400,
            content={"message": "set refer audio failed", "Exception": str(error)},
        )
    return JSONResponse(status_code=200, content={"message": "success"})


async def set_gpt_weights(weights_path: str = None):
    """加载匹配当前 SoVITS 版本的 GPT 权重；只影响当前进程。"""

    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        async with service_app.state.pipeline_lock:
            await run_in_threadpool_shielded(
                service_app.state.tts_pipeline.init_t2s_weights, weights_path
            )
    except Exception as error:
        return JSONResponse(
            status_code=400,
            content={"message": "change gpt weight failed", "Exception": str(error)},
        )
    return JSONResponse(status_code=200, content={"message": "success"})


async def set_sovits_weights(weights_path: str = None):
    """加载 SoVITS 权重及对应版本。切换整套声音时，随后加载匹配 GPT 和参考音频。"""

    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        async with service_app.state.pipeline_lock:
            await run_in_threadpool_shielded(
                service_app.state.tts_pipeline.init_vits_weights, weights_path
            )
    except Exception as error:
        return JSONResponse(
            status_code=400,
            content={"message": "change sovits weight failed", "Exception": str(error)},
        )
    return JSONResponse(status_code=200, content={"message": "success"})


def register_routes(application: FastAPI) -> None:
    """Attach the stable inference routes to an application instance."""

    application.add_api_route("/tts", tts_get_endpoint, methods=["GET"])
    application.add_api_route("/tts", tts_post_endpoint, methods=["POST"])
    application.add_api_route("/set_refer_audio", set_refer_aduio, methods=["GET"])
    application.add_api_route("/set_gpt_weights", set_gpt_weights, methods=["GET"])
    application.add_api_route("/set_sovits_weights", set_sovits_weights, methods=["GET"])
