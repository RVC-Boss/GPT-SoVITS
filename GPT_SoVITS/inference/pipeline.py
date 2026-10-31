"""Canonical stateful TTS pipeline assembled from inference mixins."""

from __future__ import annotations

from typing import Union

import torch

from GPT_SoVITS.runtime.portable import prepare_transformers_flash_attention

prepare_transformers_flash_attention()

from transformers import AutoModel, AutoTokenizer

from GPT_SoVITS.AR.models.t2s_inference import Text2SemanticInference
from GPT_SoVITS.feature_extractor.cnhubert import CNHubert
from GPT_SoVITS.inference.super_resolution import AP_BWE
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3

from .batching import BatchingMixin
from .config import TTSConfig
from .model_runtime import ModelRuntimeMixin
from .reference import ReferenceMixin
from .synthesis import SynthesisMixin
from .text_frontend import TextPreprocessor


class TTS(ModelRuntimeMixin, ReferenceMixin, BatchingMixin, SynthesisMixin):
    def __init__(self, configs: Union[dict, str, TTSConfig]):
        if isinstance(configs, TTSConfig):
            self.configs = configs
        else:
            self.configs: TTSConfig = TTSConfig(configs)

        self.t2s_model: Text2SemanticInference = None
        self.t2s_accel = None
        self.streaming_accel = None
        self.vits_model: Union[SynthesizerTrn, SynthesizerTrnV3] = None
        self.bert_tokenizer: AutoTokenizer = None
        self.bert_model: AutoModel = None
        self.cnhuhbert_model: CNHubert = None
        self.vocoder = None
        self.sr_model: AP_BWE = None
        self.sv_model = None
        self.sr_model_not_exist: bool = False

        self.vocoder_configs: dict = {
            "sr": None,
            "T_ref": None,
            "T_chunk": None,
            "upsample_rate": None,
            "overlapped_len": None,
        }

        self._init_models()

        self.text_preprocessor: TextPreprocessor = TextPreprocessor(
            self.bert_model, self.bert_tokenizer, self.configs.device
        )

        self.prompt_cache: dict = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
            "aux_ref_audio_paths": [],
        }

        self.stop_flag: bool = False
        self.precision: torch.dtype = torch.float16 if self.configs.is_half else torch.float32

    def stop(self):
        """Stop the inference process."""
        self.stop_flag = True

TTS_Config = TTSConfig
