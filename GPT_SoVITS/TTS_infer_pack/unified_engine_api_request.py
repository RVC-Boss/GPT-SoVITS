from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import NormalizedEngineRequest, ReferenceRegistry


def normalize_lang(value: str | None) -> str | None:
    if value in [None, ""]:
        return value
    return str(value).lower()


def apply_default_reference(reference_registry: ReferenceRegistry, req: dict) -> dict:
    normalized = dict(req)
    default_ref = reference_registry.get_default()
    if normalized.get("ref_audio_path") in [None, ""] and default_ref.ref_audio_path not in [None, ""]:
        normalized["ref_audio_path"] = default_ref.ref_audio_path
    if "text_lang" in normalized:
        normalized["text_lang"] = normalize_lang(normalized.get("text_lang"))
    if "prompt_lang" in normalized:
        normalized["prompt_lang"] = normalize_lang(normalized.get("prompt_lang"))
    return normalized


def check_params(tts: TTS, cut_method_names: Sequence[str], req: dict) -> Optional[str]:
    text = req.get("text", "")
    text_lang = req.get("text_lang", "")
    ref_audio_path = req.get("ref_audio_path", "")
    media_type = req.get("media_type", "wav")
    prompt_lang = req.get("prompt_lang", "")
    text_split_method = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return "ref_audio_path is required"
    if text in [None, ""]:
        return "text is required"
    if text_lang in [None, ""]:
        return "text_lang is required"
    if text_lang.lower() not in tts.configs.languages:
        return f"text_lang: {text_lang} is not supported in version {tts.configs.version}"
    if prompt_lang in [None, ""]:
        return "prompt_lang is required"
    if prompt_lang.lower() not in tts.configs.languages:
        return f"prompt_lang: {prompt_lang} is not supported in version {tts.configs.version}"
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return f"media_type: {media_type} is not supported"
    if text_split_method not in cut_method_names:
        return f"text_split_method:{text_split_method} is not supported"
    return None


def base_request_defaults() -> Dict[str, Any]:
    return {
        "request_id": None,
        "text": None,
        "text_lang": None,
        "ref_audio_path": None,
        "aux_ref_audio_paths": None,
        "prompt_text": "",
        "prompt_lang": None,
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut5",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "speed_factor": 1.0,
        "split_bucket": False,
        "fragment_interval": 0.3,
        "seed": -1,
        "media_type": "wav",
        "streaming_mode": False,
        "return_fragment": False,
        "fixed_length_chunk": False,
        "response_streaming": False,
        "parallel_infer": False,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "super_sampling": False,
        "overlap_length": 2,
        "min_chunk_length": 16,
        "early_stop_num": -1,
        "ready_step": 0,
        "timeout_sec": None,
    }


def normalize_streaming_mode(req: dict) -> dict:
    normalized = dict(req)
    streaming_mode = normalized.get("streaming_mode", False)
    return_fragment = normalized.get("return_fragment", False)
    if streaming_mode is False:
        normalized["streaming_mode"] = False
        normalized["return_fragment"] = False
        normalized["fixed_length_chunk"] = False
    elif streaming_mode == 0:
        normalized["streaming_mode"] = False
        normalized["return_fragment"] = False
        normalized["fixed_length_chunk"] = False
    elif streaming_mode == 1 or streaming_mode is True:
        normalized["streaming_mode"] = False
        normalized["return_fragment"] = True
        normalized["fixed_length_chunk"] = False
    elif streaming_mode == 2:
        normalized["streaming_mode"] = True
        normalized["return_fragment"] = False
        normalized["fixed_length_chunk"] = False
    elif streaming_mode == 3:
        normalized["streaming_mode"] = True
        normalized["return_fragment"] = False
        normalized["fixed_length_chunk"] = True
    else:
        raise ValueError("the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)")
    normalized["response_streaming"] = bool(normalized["streaming_mode"] or normalized["return_fragment"] or return_fragment)
    return normalized


def is_aux_ref_enabled(aux_ref_audio_paths: List[str] | None) -> bool:
    return aux_ref_audio_paths not in [None, [], ()]


def select_direct_backend(normalized: NormalizedEngineRequest) -> Tuple[str, str | None]:
    return "scheduler_v1_direct", None


def normalize_engine_request(
    *,
    tts: TTS,
    cut_method_names: Sequence[str],
    reference_registry: ReferenceRegistry,
    payload: dict | NormalizedEngineRequest,
    request_id: str | None = None,
    normalize_streaming: bool = False,
    error_prefix: str = "request 参数非法: ",
) -> NormalizedEngineRequest:
    if isinstance(payload, NormalizedEngineRequest):
        normalized_payload = payload.to_payload()
    else:
        normalized_payload = base_request_defaults()
        normalized_payload.update(dict(payload))
    if request_id not in [None, ""]:
        normalized_payload["request_id"] = str(request_id)
    elif normalized_payload.get("request_id") in [None, ""]:
        raise ValueError("request_id is required after normalization")
    normalized_payload = apply_default_reference(reference_registry, normalized_payload)
    if normalize_streaming:
        normalized_payload = normalize_streaming_mode(normalized_payload)
    error = check_params(tts, cut_method_names, normalized_payload)
    if error is not None:
        raise ValueError(f"{error_prefix}{error}")
    timeout_sec = normalized_payload.get("timeout_sec")
    parsed_timeout = None if timeout_sec in [None, ""] else float(timeout_sec)
    aux_ref_audio_paths = normalized_payload.get("aux_ref_audio_paths")
    normalized_aux_ref_audio_paths = None if aux_ref_audio_paths in [None, "", []] else [str(item) for item in aux_ref_audio_paths]
    return NormalizedEngineRequest(
        request_id=str(normalized_payload["request_id"]),
        text=str(normalized_payload["text"]),
        text_lang=str(normalized_payload["text_lang"]),
        ref_audio_path=str(normalized_payload["ref_audio_path"]),
        prompt_lang=str(normalized_payload["prompt_lang"]),
        prompt_text="" if normalized_payload.get("prompt_text") is None else str(normalized_payload.get("prompt_text")),
        aux_ref_audio_paths=normalized_aux_ref_audio_paths,
        top_k=int(normalized_payload["top_k"]),
        top_p=float(normalized_payload["top_p"]),
        temperature=float(normalized_payload["temperature"]),
        repetition_penalty=float(normalized_payload["repetition_penalty"]),
        early_stop_num=int(normalized_payload.get("early_stop_num", -1)),
        ready_step=int(normalized_payload.get("ready_step", 0)),
        text_split_method=str(normalized_payload["text_split_method"]),
        batch_size=int(normalized_payload["batch_size"]),
        batch_threshold=float(normalized_payload["batch_threshold"]),
        split_bucket=bool(normalized_payload["split_bucket"]),
        speed_factor=float(normalized_payload["speed_factor"]),
        fragment_interval=float(normalized_payload["fragment_interval"]),
        seed=int(normalized_payload["seed"]),
        media_type=str(normalized_payload["media_type"]),
        streaming_mode=normalized_payload["streaming_mode"],
        return_fragment=bool(normalized_payload.get("return_fragment", False)),
        fixed_length_chunk=bool(normalized_payload.get("fixed_length_chunk", False)),
        response_streaming=bool(normalized_payload.get("response_streaming", False)),
        parallel_infer=bool(normalized_payload["parallel_infer"]),
        sample_steps=int(normalized_payload["sample_steps"]),
        super_sampling=bool(normalized_payload["super_sampling"]),
        overlap_length=int(normalized_payload["overlap_length"]),
        min_chunk_length=int(normalized_payload["min_chunk_length"]),
        timeout_sec=parsed_timeout,
    )
