from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec


@dataclass
class RuntimeControlCallbacks:
    restart: Callable[[], None] | None = None
    exit: Callable[[], None] | None = None


@dataclass
class DirectTTSExecution:
    media_type: str
    streaming: bool
    audio_generator: Optional[Generator[bytes, None, None]] = None
    audio_bytes: Optional[bytes] = None
    request_id: Optional[str] = None


@dataclass
class NormalizedEngineRequest:
    request_id: str
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_lang: str
    prompt_text: str = ""
    aux_ref_audio_paths: List[str] | None = None
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    early_stop_num: int = -1
    ready_step: int = 0
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = False
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool | int = False
    return_fragment: bool = False
    fixed_length_chunk: bool = False
    response_streaming: bool = False
    parallel_infer: bool = False
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    timeout_sec: float | None = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text": self.text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "aux_ref_audio_paths": list(self.aux_ref_audio_paths) if self.aux_ref_audio_paths else None,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "batch_threshold": self.batch_threshold,
            "speed_factor": self.speed_factor,
            "split_bucket": self.split_bucket,
            "fragment_interval": self.fragment_interval,
            "seed": self.seed,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,
            "return_fragment": self.return_fragment,
            "fixed_length_chunk": self.fixed_length_chunk,
            "response_streaming": self.response_streaming,
            "parallel_infer": self.parallel_infer,
            "repetition_penalty": self.repetition_penalty,
            "sample_steps": self.sample_steps,
            "super_sampling": self.super_sampling,
            "overlap_length": self.overlap_length,
            "min_chunk_length": self.min_chunk_length,
            "early_stop_num": self.early_stop_num,
            "ready_step": self.ready_step,
            "timeout_sec": self.timeout_sec,
        }

    def to_scheduler_spec(self) -> SchedulerRequestSpec:
        return SchedulerRequestSpec(
            request_id=self.request_id,
            ref_audio_path=Path(self.ref_audio_path),
            prompt_text=self.prompt_text,
            prompt_lang=self.prompt_lang,
            text=self.text,
            text_lang=self.text_lang,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            early_stop_num=self.early_stop_num,
            ready_step=self.ready_step,
        )


@dataclass
class SchedulerDebugExecution:
    payload: Dict[str, Any]


@dataclass
class SchedulerSubmitExecution:
    audio_bytes: bytes
    media_type: str
    headers: Dict[str, str]
