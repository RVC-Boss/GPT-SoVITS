"""Configuration contract for the stateful TTS pipeline."""

import os
from copy import deepcopy
from typing import Union

import torch
import yaml

from GPT_SoVITS.runtime.i18n import I18nAuto

language = os.environ.get("language", "Auto")
i18n = I18nAuto(language=language)
V5_VERSIONS = {"v5dev", "v5turbo"}


class TTSConfig:
    default_configs = {
        "v1": {
            "device": "cpu",
            "is_half": False,
            "version": "v1",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/s2G488k.pth",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "v2": {
            "device": "cpu",
            "is_half": False,
            "version": "v2",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "v3": {
            "device": "cpu",
            "is_half": False,
            "version": "v3",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "v4": {
            "device": "cpu",
            "is_half": False,
            "version": "v4",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "v2Pro": {
            "device": "cpu",
            "is_half": False,
            "version": "v2Pro",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "v2ProPlus": {
            "device": "cpu",
            "is_half": False,
            "version": "v2ProPlus",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
    }
    for v5_variant in V5_VERSIONS:
        default_configs[v5_variant] = dict(
            default_configs["v4"], version=v5_variant,
            vits_weights_path=f"GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2G{v5_variant}.pth",
        )
    configs: dict = None
    v1_languages: list = ["auto", "en", "zh", "ja", "all_zh", "all_ja"]
    v2_languages: list = ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
    languages: list = v2_languages
    mute_tokens: dict = {
        "v1" : 486,
        "v2" : 486,
        "v2Pro": 486,
        "v2ProPlus": 486,
        "v3" : 486,
        "v4" : 486,
        "v5dev": 486,
        "v5turbo": 486,
    }
    mute_emb_sim_matrix: torch.Tensor = None
    # "all_zh",#全部按中文识别
    # "en",#全部按英文识别#######不变
    # "all_ja",#全部按日文识别
    # "all_yue",#全部按中文识别
    # "all_ko",#全部按韩文识别
    # "zh",#按中英混合识别####不变
    # "ja",#按日英混合识别####不变
    # "yue",#按粤英混合识别####不变
    # "ko",#按韩英混合识别####不变
    # "auto",#多语种启动切分识别语种
    # "auto_yue",#多语种启动切分识别语种

    def __init__(self, configs: Union[dict, str] = None):
        # 设置默认配置文件路径
        configs_base_path: str = "GPT_SoVITS/configs/"
        self.configs_path: str = os.path.join(configs_base_path, "tts_infer.yaml")

        if configs in ["", None]:
            configs: dict = deepcopy(self.default_configs)

        if isinstance(configs, str):
            self.configs_path = configs
            configs: dict = self._load_configs(self.configs_path)

        assert isinstance(configs, dict)
        configs_ = deepcopy(self.default_configs)
        configs_.update(configs)
        self.configs: dict = configs_.get("custom", configs_["v2"])
        self.default_configs = deepcopy(configs_)

        self.device = self.configs.get("device", torch.device("cpu"))
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            print("Warning: CUDA is not available, set device to CPU.")
            self.device = torch.device("cpu")

        self.is_half = self.configs.get("is_half", False)
        if str(self.device) == "cpu" and self.is_half:
            print(f"Warning: Half precision is not supported on CPU, set is_half to False.")
            self.is_half = False

        version = self.configs.get("version", None)
        self.version = version
        assert self.version in ["v1", "v2", "v3", "v4", "v5dev", "v5turbo", "v2Pro", "v2ProPlus"], "Invalid version!"
        self.t2s_weights_path = self.configs.get("t2s_weights_path", None)
        self.vits_weights_path = self.configs.get("vits_weights_path", None)
        self.bert_base_path = self.configs.get("bert_base_path", None)
        self.cnhuhbert_base_path = self.configs.get("cnhuhbert_base_path", None)
        self.languages = self.v1_languages if self.version == "v1" else self.v2_languages

        self.use_vocoder: bool = False

        if (self.t2s_weights_path in [None, ""]) or (not os.path.exists(self.t2s_weights_path)):
            self.t2s_weights_path = self.default_configs[version]["t2s_weights_path"]
            print(f"fall back to default t2s_weights_path: {self.t2s_weights_path}")
        if (self.vits_weights_path in [None, ""]) or (not os.path.exists(self.vits_weights_path)):
            self.vits_weights_path = self.default_configs[version]["vits_weights_path"]
            print(f"fall back to default vits_weights_path: {self.vits_weights_path}")
        if (self.bert_base_path in [None, ""]) or (not os.path.exists(self.bert_base_path)):
            self.bert_base_path = self.default_configs[version]["bert_base_path"]
            print(f"fall back to default bert_base_path: {self.bert_base_path}")
        if (self.cnhuhbert_base_path in [None, ""]) or (not os.path.exists(self.cnhuhbert_base_path)):
            self.cnhuhbert_base_path = self.default_configs[version]["cnhuhbert_base_path"]
            print(f"fall back to default cnhuhbert_base_path: {self.cnhuhbert_base_path}")
        self.update_configs()

        self.max_sec = None
        self.hz: int = 50
        self.semantic_frame_rate: str = "25hz"
        self.filter_length: int = 2048
        self.sampling_rate: int = 32000
        self.hop_length: int = 640
        self.win_length: int = 2048

    def _load_configs(self, configs_path: str) -> dict:
        if not os.path.exists(configs_path):
            print(i18n("路径不存在,使用默认配置"))
            return deepcopy(self.default_configs)
        with open(configs_path, "r", encoding="utf-8") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        return configs

    def update_configs(self):
        self.config = {
            "device": str(self.device),
            "is_half": self.is_half,
            "version": self.version,
            "t2s_weights_path": self.t2s_weights_path,
            "vits_weights_path": self.vits_weights_path,
            "bert_base_path": self.bert_base_path,
            "cnhuhbert_base_path": self.cnhuhbert_base_path,
        }
        return self.config

    def update_version(self, version: str) -> None:
        self.version = version
        self.languages = self.v1_languages if self.version == "v1" else self.v2_languages

    def __str__(self):
        self.configs = self.update_configs()
        string = "TTS Config".center(100, "-") + "\n"
        for k, v in self.configs.items():
            string += f"{str(k).ljust(20)}: {str(v)}\n"
        string += "-" * 100 + "\n"
        return string

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.configs_path)

    def __eq__(self, other):
        return isinstance(other, TTSConfig) and self.configs_path == other.configs_path


TTS_Config = TTSConfig
