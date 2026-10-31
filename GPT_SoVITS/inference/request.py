"""Normalized options for direct pipeline synthesis calls."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class SynthesisRequest:
    text: str = ""
    text_lang: str = ""
    ref_audio_path: str = ""
    aux_ref_audio_paths: tuple[str, ...] = ()
    prompt_text: str = ""
    prompt_lang: str = ""
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut1"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    parallel_infer: bool = True
    use_cuda_graph: bool = True
    use_flash_attention: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    cfg_rate: float = 0.0
    super_sampling: bool = False
    return_fragment: bool = False
    streaming_mode: bool | int = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    fixed_length_chunk: bool = False
    gpt_streaming: bool = False
    streaming_chunk_seconds: float = 2.0
    streaming_window_mode: str = "fixed"
    streaming_lookahead_seconds: float = 0.12
    streaming_overlap_seconds: float = 0.04
    gpt_streaming_lookahead_seconds: float = 0.5
    media_type: str = "wav"

    @classmethod
    def from_mapping(cls, inputs: Mapping[str, Any], version: str) -> "SynthesisRequest":
        values = dict(inputs)
        values["aux_ref_audio_paths"] = tuple(values.get("aux_ref_audio_paths") or ())
        values["seed"] = -1 if values.get("seed", -1) in ("", None) else int(values.get("seed", -1))
        sample_steps = values.get("sample_steps")
        values["sample_steps"] = (4 if version == "v5turbo" else 32) if sample_steps is None else int(sample_steps)
        cfg_rate = values.get("cfg_rate")
        values["cfg_rate"] = (1.30 if version == "v5dev" else 0.0) if cfg_rate is None else float(cfg_rate)
        known = cls.__dataclass_fields__
        return cls(**{name: values[name] for name in known if name in values})

    def as_dict(self) -> dict[str, Any]:
        values = dict(self.__dict__)
        values["aux_ref_audio_paths"] = list(self.aux_ref_audio_paths)
        return values


def set_seed(seed: int) -> int:
    seed = int(seed)
    seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
    print(f"Set seed to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    return seed
