"""Public request schema and HTTP-to-pipeline normalization."""

from typing import Annotated, Optional, Union

from fastapi import Query, Request
from pydantic import BaseModel


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


def request_model_to_dict(request: TTS_Request) -> dict:
    """Convert either supported Pydantic generation to a plain request dict."""

    return request.model_dump() if hasattr(request, "model_dump") else request.dict()


def get_request_dict(
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
) -> dict:
    """Build the internal request mapping while preserving GET coercions."""

    return {
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


def raw_get_streaming_mode(request: Request):
    """Retain the uncoerced mode needed by semantic-stream validation."""

    raw_mode = request.query_params.get("streaming_mode")
    if raw_mode in {"2", "3"}:
        return int(raw_mode)
    return raw_mode
