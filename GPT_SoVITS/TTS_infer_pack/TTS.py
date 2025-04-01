import gc
import math
import os
import random
import sys
import time
import traceback
from copy import deepcopy

import torchaudio
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)
import os
from typing import List, Tuple, Union

import ffmpeg
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from BigVGAN.bigvgan import BigVGAN
from feature_extractor.cnhubert import CNHubert
from module.mel_processing import mel_spectrogram_torch, spectrogram_torch
from module.models import SynthesizerTrn, SynthesizerTrnV3
from peft import LoraConfig, get_peft_model
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from transformers import AutoModelForMaskedLM, AutoTokenizer

from tools.audio_sr import AP_BWE
from tools.i18n.i18n import I18nAuto, scan_language_list
from tools.my_utils import load_audio
from TTS_infer_pack.text_segmentation_method import splits
from TTS_infer_pack.TextPreprocessor import TextPreprocessor

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)


spec_min = -12
spec_max = 2


def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)


def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    # 将 NumPy 数组转换为原始 PCM 流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input("pipe:", format="s16le", acodec="pcm_s16le", ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter("atempo", speed)

    # 输出流到管道
    out, _ = output_stream.output("pipe:", format="s16le", acodec="pcm_s16le").run(
        input=raw_audio, capture_stdout=True, capture_stderr=True
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio


resample_transform_dict = {}


def resample(audio_tensor, sr0, device):
    global resample_transform_dict
    if sr0 not in resample_transform_dict:
        resample_transform_dict[sr0] = torchaudio.transforms.Resample(sr0, 24000).to(device)
    return resample_transform_dict[sr0](audio_tensor)


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class NO_PROMPT_ERROR(Exception):
    pass


# configs/tts_infer.yaml
"""
custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
  version: v2
default:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/s2G488k.pth
  version: v1
default_v2:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
  version: v2
default_v3:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1v3.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/s2Gv3.pth
  version: v3
"""


def set_seed(seed: int):
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
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.enabled = True
            # 开启后会影响精度
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    return seed


class TTS_Config:
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
    }
    configs: dict = None
    v1_languages: list = ["auto", "en", "zh", "ja", "all_zh", "all_ja"]
    v2_languages: list = ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
    languages: list = v2_languages
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
        os.makedirs(configs_base_path, exist_ok=True)
        self.configs_path: str = os.path.join(configs_base_path, "tts_infer.yaml")

        if configs in ["", None]:
            if not os.path.exists(self.configs_path):
                self.save_configs()
                print(f"Create default config file at {self.configs_path}")
            configs: dict = deepcopy(self.default_configs)

        if isinstance(configs, str):
            self.configs_path = configs
            configs: dict = self._load_configs(self.configs_path)

        assert isinstance(configs, dict)
        version = configs.get("version", "v2").lower()
        assert version in ["v1", "v2", "v3"]
        self.default_configs[version] = configs.get(version, self.default_configs[version])
        self.configs: dict = configs.get("custom", deepcopy(self.default_configs[version]))

        self.device = self.configs.get("device", torch.device("cpu"))
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            print("Warning: CUDA is not available, set device to CPU.")
            self.device = torch.device("cpu")

        self.is_half = self.configs.get("is_half", False)
        # if str(self.device) == "cpu" and self.is_half:
        #     print(f"Warning: Half precision is not supported on CPU, set is_half to False.")
        #     self.is_half = False

        self.version = version
        self.t2s_weights_path = self.configs.get("t2s_weights_path", None)
        self.vits_weights_path = self.configs.get("vits_weights_path", None)
        self.bert_base_path = self.configs.get("bert_base_path", None)
        self.cnhuhbert_base_path = self.configs.get("cnhuhbert_base_path", None)
        self.languages = self.v1_languages if self.version == "v1" else self.v2_languages

        self.is_v3_synthesizer: bool = False

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
        self.segment_size: int = 20480
        self.filter_length: int = 2048
        self.sampling_rate: int = 32000
        self.hop_length: int = 640
        self.win_length: int = 2048
        self.n_speakers: int = 300

    def _load_configs(self, configs_path: str) -> dict:
        if os.path.exists(configs_path):
            ...
        else:
            print(i18n("路径不存在,使用默认配置"))
            self.save_configs(configs_path)
        with open(configs_path, "r", encoding="utf-8") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        return configs

    def save_configs(self, configs_path: str = None) -> None:
        configs = deepcopy(self.default_configs)
        if self.configs is not None:
            configs["custom"] = self.update_configs()

        if configs_path is None:
            configs_path = self.configs_path
        with open(configs_path, "w") as f:
            yaml.dump(configs, f)

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
        return isinstance(other, TTS_Config) and self.configs_path == other.configs_path


class TTS:
    def __init__(self, configs: Union[dict, str, TTS_Config]):
        if isinstance(configs, TTS_Config):
            self.configs = configs
        else:
            self.configs: TTS_Config = TTS_Config(configs)

        self.t2s_model: Text2SemanticLightningModule = None
        self.vits_model: Union[SynthesizerTrn, SynthesizerTrnV3] = None
        self.bert_tokenizer: AutoTokenizer = None
        self.bert_model: AutoModelForMaskedLM = None
        self.cnhuhbert_model: CNHubert = None
        self.bigvgan_model: BigVGAN = None
        self.sr_model: AP_BWE = None
        self.sr_model_not_exist: bool = False

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

    def _init_models(
        self,
    ):
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_vits_weights(self.configs.vits_weights_path)
        self.init_bert_weights(self.configs.bert_base_path)
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)
        # self.enable_half_precision(self.configs.is_half)

    def init_cnhuhbert_weights(self, base_path: str):
        print(f"Loading CNHuBERT weights from {base_path}")
        self.cnhuhbert_model = CNHubert(base_path)
        self.cnhuhbert_model = self.cnhuhbert_model.eval()
        self.cnhuhbert_model = self.cnhuhbert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.cnhuhbert_model = self.cnhuhbert_model.half()

    def init_bert_weights(self, base_path: str):
        print(f"Loading BERT weights from {base_path}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
        self.bert_model = self.bert_model.eval()
        self.bert_model = self.bert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.bert_model = self.bert_model.half()

    def init_vits_weights(self, weights_path: str):
        self.configs.vits_weights_path = weights_path
        version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(weights_path)
        path_sovits_v3 = self.configs.default_configs["v3"]["vits_weights_path"]

        if if_lora_v3 == True and os.path.exists(path_sovits_v3) == False:
            info = path_sovits_v3 + i18n("SoVITS V3 底模缺失，无法加载相应 LoRA 权重")
            raise FileExistsError(info)

        # dict_s2 = torch.load(weights_path, map_location=self.configs.device,weights_only=False)
        dict_s2 = load_sovits_new(weights_path)
        hps = dict_s2["config"]

        hps["model"]["semantic_frame_rate"] = "25hz"
        if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
            hps["model"]["version"] = "v2"  # v3model,v2sybomls
        elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps["model"]["version"] = "v1"
        else:
            hps["model"]["version"] = "v2"
        # version = hps["model"]["version"]

        self.configs.filter_length = hps["data"]["filter_length"]
        self.configs.segment_size = hps["train"]["segment_size"]
        self.configs.sampling_rate = hps["data"]["sampling_rate"]
        self.configs.hop_length = hps["data"]["hop_length"]
        self.configs.win_length = hps["data"]["win_length"]
        self.configs.n_speakers = hps["data"]["n_speakers"]
        self.configs.semantic_frame_rate = hps["model"]["semantic_frame_rate"]
        kwargs = hps["model"]
        # print(f"self.configs.sampling_rate:{self.configs.sampling_rate}")

        self.configs.update_version(model_version)

        # print(f"model_version:{model_version}")
        # print(f'hps["model"]["version"]:{hps["model"]["version"]}')
        if model_version != "v3":
            vits_model = SynthesizerTrn(
                self.configs.filter_length // 2 + 1,
                self.configs.segment_size // self.configs.hop_length,
                n_speakers=self.configs.n_speakers,
                **kwargs,
            )
            self.configs.is_v3_synthesizer = False
        else:
            vits_model = SynthesizerTrnV3(
                self.configs.filter_length // 2 + 1,
                self.configs.segment_size // self.configs.hop_length,
                n_speakers=self.configs.n_speakers,
                **kwargs,
            )
            self.configs.is_v3_synthesizer = True
            self.init_bigvgan()
            if "pretrained" not in weights_path and hasattr(vits_model, "enc_q"):
                del vits_model.enc_q

        if if_lora_v3 == False:
            print(
                f"Loading VITS weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}"
            )
        else:
            print(
                f"Loading VITS pretrained weights from {weights_path}. {vits_model.load_state_dict(load_sovits_new(path_sovits_v3)['weight'], strict=False)}"
            )
            lora_rank = dict_s2["lora_rank"]
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            vits_model.cfm = get_peft_model(vits_model.cfm, lora_config)
            print(
                f"Loading LoRA weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}"
            )

            vits_model.cfm = vits_model.cfm.merge_and_unload()

        vits_model = vits_model.to(self.configs.device)
        vits_model = vits_model.eval()

        self.vits_model = vits_model
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.vits_model = self.vits_model.half()

    def init_t2s_weights(self, weights_path: str):
        print(f"Loading Text2Semantic weights from {weights_path}")
        self.configs.t2s_weights_path = weights_path
        self.configs.save_configs()
        self.configs.hz = 50
        dict_s1 = torch.load(weights_path, map_location=self.configs.device)
        config = dict_s1["config"]
        self.configs.max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(self.configs.device)
        t2s_model = t2s_model.eval()
        self.t2s_model = t2s_model
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.t2s_model = self.t2s_model.half()

    def init_bigvgan(self):
        if self.bigvgan_model is not None:
            return
        self.bigvgan_model = BigVGAN.from_pretrained(
            "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
            use_cuda_kernel=False,
        )  # if True, RuntimeError: Ninja is required to load C++ extensions
        # remove weight norm in the model and set to eval mode
        self.bigvgan_model.remove_weight_norm()
        self.bigvgan_model = self.bigvgan_model.eval()
        if self.configs.is_half == True:
            self.bigvgan_model = self.bigvgan_model.half().to(self.configs.device)
        else:
            self.bigvgan_model = self.bigvgan_model.to(self.configs.device)

    def init_sr_model(self):
        if self.sr_model is not None:
            return
        try:
            self.sr_model: AP_BWE = AP_BWE(self.configs.device, DictToAttrRecursive)
            self.sr_model_not_exist = False
        except FileNotFoundError:
            print(i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
            self.sr_model_not_exist = True

    def enable_half_precision(self, enable: bool = True, save: bool = True):
        """
        To enable half precision for the TTS model.
        Args:
            enable: bool, whether to enable half precision.

        """
        if str(self.configs.device) == "cpu" and enable:
            print("Half precision is not supported on CPU.")
            return

        self.configs.is_half = enable
        self.precision = torch.float16 if enable else torch.float32
        if save:
            self.configs.save_configs()
        if enable:
            if self.t2s_model is not None:
                self.t2s_model = self.t2s_model.half()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.half()
            if self.bert_model is not None:
                self.bert_model = self.bert_model.half()
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.half()
            if self.bigvgan_model is not None:
                self.bigvgan_model = self.bigvgan_model.half()
        else:
            if self.t2s_model is not None:
                self.t2s_model = self.t2s_model.float()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.float()
            if self.bert_model is not None:
                self.bert_model = self.bert_model.float()
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.float()
            if self.bigvgan_model is not None:
                self.bigvgan_model = self.bigvgan_model.float()

    def set_device(self, device: torch.device, save: bool = True):
        """
        To set the device for all models.
        Args:
            device: torch.device, the device to use for all models.
        """
        self.configs.device = device
        if save:
            self.configs.save_configs()
        if self.t2s_model is not None:
            self.t2s_model = self.t2s_model.to(device)
        if self.vits_model is not None:
            self.vits_model = self.vits_model.to(device)
        if self.bert_model is not None:
            self.bert_model = self.bert_model.to(device)
        if self.cnhuhbert_model is not None:
            self.cnhuhbert_model = self.cnhuhbert_model.to(device)
        if self.bigvgan_model is not None:
            self.bigvgan_model = self.bigvgan_model.to(device)
        if self.sr_model is not None:
            self.sr_model = self.sr_model.to(device)

    def set_ref_audio(self, ref_audio_path: str):
        """
        To set the reference audio for the TTS model,
            including the prompt_semantic and refer_spepc.
        Args:
            ref_audio_path: str, the path of the reference audio.
        """
        self._set_prompt_semantic(ref_audio_path)
        self._set_ref_spec(ref_audio_path)
        self._set_ref_audio_path(ref_audio_path)

    def _set_ref_audio_path(self, ref_audio_path):
        self.prompt_cache["ref_audio_path"] = ref_audio_path

    def _set_ref_spec(self, ref_audio_path):
        spec = self._get_ref_spec(ref_audio_path)
        if self.prompt_cache["refer_spec"] in [[], None]:
            self.prompt_cache["refer_spec"] = [spec]
        else:
            self.prompt_cache["refer_spec"][0] = spec

    def _get_ref_spec(self, ref_audio_path):
        raw_audio, raw_sr = torchaudio.load(ref_audio_path)
        raw_audio = raw_audio.to(self.configs.device).float()
        self.prompt_cache["raw_audio"] = raw_audio
        self.prompt_cache["raw_sr"] = raw_sr

        audio = load_audio(ref_audio_path, int(self.configs.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.configs.filter_length,
            self.configs.sampling_rate,
            self.configs.hop_length,
            self.configs.win_length,
            center=False,
        )
        spec = spec.to(self.configs.device)
        if self.configs.is_half:
            spec = spec.half()
        return spec

    def _set_prompt_semantic(self, ref_wav_path: str):
        zero_wav = np.zeros(
            int(self.configs.sampling_rate * 0.3),
            dtype=np.float16 if self.configs.is_half else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(self.configs.device)
            zero_wav_torch = zero_wav_torch.to(self.configs.device)
            if self.configs.is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()

            wav16k = torch.cat([wav16k, zero_wav_torch])
            hubert_feature = self.cnhuhbert_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(
                1, 2
            )  # .float()
            codes = self.vits_model.extract_latent(hubert_feature)

            prompt_semantic = codes[0, 0].to(self.configs.device)
            self.prompt_cache["prompt_semantic"] = prompt_semantic

    def batch_sequences(self, sequences: List[torch.Tensor], axis: int = 0, pad_value: int = 0, max_length: int = None):
        seq = sequences[0]
        ndim = seq.dim()
        if axis < 0:
            axis += ndim
        dtype: torch.dtype = seq.dtype
        pad_value = torch.tensor(pad_value, dtype=dtype)
        seq_lengths = [seq.shape[axis] for seq in sequences]
        if max_length is None:
            max_length = max(seq_lengths)
        else:
            max_length = max(seq_lengths) if max_length < max(seq_lengths) else max_length

        padded_sequences = []
        for seq, length in zip(sequences, seq_lengths):
            padding = [0] * axis + [0, max_length - length] + [0] * (ndim - axis - 1)
            padded_seq = torch.nn.functional.pad(seq, padding, value=pad_value)
            padded_sequences.append(padded_seq)
        batch = torch.stack(padded_sequences)
        return batch

    def to_batch(
        self,
        data: list,
        prompt_data: dict = None,
        batch_size: int = 5,
        threshold: float = 0.75,
        split_bucket: bool = True,
        device: torch.device = torch.device("cpu"),
        precision: torch.dtype = torch.float32,
    ):
        _data: list = []
        index_and_len_list = []
        for idx, item in enumerate(data):
            norm_text_len = len(item["norm_text"])
            index_and_len_list.append([idx, norm_text_len])

        batch_index_list = []
        if split_bucket:
            index_and_len_list.sort(key=lambda x: x[1])
            index_and_len_list = np.array(index_and_len_list, dtype=np.int64)

            batch_index_list_len = 0
            pos = 0
            while pos < index_and_len_list.shape[0]:
                # batch_index_list.append(index_and_len_list[pos:min(pos+batch_size,len(index_and_len_list))])
                pos_end = min(pos + batch_size, index_and_len_list.shape[0])
                while pos < pos_end:
                    batch = index_and_len_list[pos:pos_end, 1].astype(np.float32)
                    score = batch[(pos_end - pos) // 2] / (batch.mean() + 1e-8)
                    if (score >= threshold) or (pos_end - pos == 1):
                        batch_index = index_and_len_list[pos:pos_end, 0].tolist()
                        batch_index_list_len += len(batch_index)
                        batch_index_list.append(batch_index)
                        pos = pos_end
                        break
                    pos_end = pos_end - 1

            assert batch_index_list_len == len(data)

        else:
            for i in range(len(data)):
                if i % batch_size == 0:
                    batch_index_list.append([])
                batch_index_list[-1].append(i)

        for batch_idx, index_list in enumerate(batch_index_list):
            item_list = [data[idx] for idx in index_list]
            phones_list = []
            phones_len_list = []
            # bert_features_list = []
            all_phones_list = []
            all_phones_len_list = []
            all_bert_features_list = []
            norm_text_batch = []
            all_bert_max_len = 0
            all_phones_max_len = 0
            for item in item_list:
                if prompt_data is not None:
                    all_bert_features = torch.cat([prompt_data["bert_features"], item["bert_features"]], 1).to(
                        dtype=precision, device=device
                    )
                    all_phones = torch.LongTensor(prompt_data["phones"] + item["phones"]).to(device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    # norm_text = prompt_data["norm_text"]+item["norm_text"]
                else:
                    all_bert_features = item["bert_features"].to(dtype=precision, device=device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    all_phones = phones
                    # norm_text = item["norm_text"]

                all_bert_max_len = max(all_bert_max_len, all_bert_features.shape[-1])
                all_phones_max_len = max(all_phones_max_len, all_phones.shape[-1])

                phones_list.append(phones)
                phones_len_list.append(phones.shape[-1])
                all_phones_list.append(all_phones)
                all_phones_len_list.append(all_phones.shape[-1])
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])

            phones_batch = phones_list
            all_phones_batch = all_phones_list
            all_bert_features_batch = all_bert_features_list

            max_len = max(all_bert_max_len, all_phones_max_len)
            # phones_batch = self.batch_sequences(phones_list, axis=0, pad_value=0, max_length=max_len)
            #### 直接对phones和bert_features进行pad。（padding策略会影响T2S模型生成的结果，但不直接影响复读概率。影响复读概率的主要因素是mask的策略）
            # all_phones_batch = self.batch_sequences(all_phones_list, axis=0, pad_value=0, max_length=max_len)
            # all_bert_features_batch = all_bert_features_list
            # all_bert_features_batch = torch.zeros((len(all_bert_features_list), 1024, max_len), dtype=precision, device=device)
            # for idx, item in enumerate(all_bert_features_list):
            #     all_bert_features_batch[idx, :, : item.shape[-1]] = item

            # #### 先对phones进行embedding、对bert_features进行project，再pad到相同长度，（padding策略会影响T2S模型生成的结果，但不直接影响复读概率。影响复读概率的主要因素是mask的策略）
            # all_phones_list = [self.t2s_model.model.ar_text_embedding(item.to(self.t2s_model.device)) for item in all_phones_list]
            # all_phones_list = [F.pad(item,(0,0,0,max_len-item.shape[0]),value=0) for item in all_phones_list]
            # all_phones_batch = torch.stack(all_phones_list, dim=0)

            # all_bert_features_list = [self.t2s_model.model.bert_proj(item.to(self.t2s_model.device).transpose(0, 1)) for item in all_bert_features_list]
            # all_bert_features_list = [F.pad(item,(0,0,0,max_len-item.shape[0]), value=0) for item in all_bert_features_list]
            # all_bert_features_batch = torch.stack(all_bert_features_list, dim=0)

            batch = {
                "phones": phones_batch,
                "phones_len": torch.LongTensor(phones_len_list).to(device),
                "all_phones": all_phones_batch,
                "all_phones_len": torch.LongTensor(all_phones_len_list).to(device),
                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch,
                "max_len": max_len,
            }
            _data.append(batch)

        return _data, batch_index_list

    def recovery_order(self, data: list, batch_index_list: list) -> list:
        """
        Recovery the order of the audio according to the batch_index_list.

        Args:
            data (List[list(torch.Tensor)]): the out of order audio .
            batch_index_list (List[list[int]]): the batch index list.

        Returns:
            list (List[torch.Tensor]): the data in the original order.
        """
        length = len(sum(batch_index_list, []))
        _data = [None] * length
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                _data[index] = data[i][j]
        return _data

    def stop(
        self,
    ):
        """
        Stop the inference process.
        """
        self.stop_flag = True

    @torch.no_grad()
    def run(self, inputs: dict):
        """
        Text to speech inference.

        Args:
            inputs (dict):
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
                    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                    "return_fragment": False,     # bool. step by step return the audio fragment.
                    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                    "seed": -1,                   # int. random seed for reproducibility.
                    "parallel_infer": True,       # bool. whether to use parallel inference.
                    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
                    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                    "super_sampling": False,       # bool. whether to use super-sampling for audio when using VITS model V3.
                }
        returns:
            Tuple[int, np.ndarray]: sampling rate and audio data.
        """
        ########## variables initialization ###########
        self.stop_flag: bool = False
        text: str = inputs.get("text", "")
        text_lang: str = inputs.get("text_lang", "")
        ref_audio_path: str = inputs.get("ref_audio_path", "")
        aux_ref_audio_paths: list = inputs.get("aux_ref_audio_paths", [])
        prompt_text: str = inputs.get("prompt_text", "")
        prompt_lang: str = inputs.get("prompt_lang", "")
        top_k: int = inputs.get("top_k", 5)
        top_p: float = inputs.get("top_p", 1)
        temperature: float = inputs.get("temperature", 1)
        text_split_method: str = inputs.get("text_split_method", "cut0")
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        split_bucket = inputs.get("split_bucket", True)
        return_fragment = inputs.get("return_fragment", False)
        fragment_interval = inputs.get("fragment_interval", 0.3)
        seed = inputs.get("seed", -1)
        seed = -1 if seed in ["", None] else seed
        actual_seed = set_seed(seed)
        parallel_infer = inputs.get("parallel_infer", True)
        repetition_penalty = inputs.get("repetition_penalty", 1.35)
        sample_steps = inputs.get("sample_steps", 32)
        super_sampling = inputs.get("super_sampling", False)

        if parallel_infer:
            print(i18n("并行推理模式已开启"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer
        else:
            print(i18n("并行推理模式已关闭"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched

        if return_fragment:
            print(i18n("分段返回模式已开启"))
            if split_bucket:
                split_bucket = False
                print(i18n("分段返回模式不支持分桶处理，已自动关闭分桶处理"))

        if split_bucket and speed_factor == 1.0 and not (self.configs.is_v3_synthesizer and parallel_infer):
            print(i18n("分桶处理模式已开启"))
        elif speed_factor != 1.0:
            print(i18n("语速调节不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        elif self.configs.is_v3_synthesizer and parallel_infer:
            print(i18n("当开启并行推理模式时，SoVits V3模型不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        else:
            print(i18n("分桶处理模式已关闭"))

        if fragment_interval < 0.01:
            fragment_interval = 0.01
            print(i18n("分段间隔过小，已自动设置为0.01"))

        no_prompt_text = False
        if prompt_text in [None, ""]:
            no_prompt_text = True

        assert text_lang in self.configs.languages
        if not no_prompt_text:
            assert prompt_lang in self.configs.languages

        if no_prompt_text and self.configs.is_v3_synthesizer:
            raise NO_PROMPT_ERROR("prompt_text cannot be empty when using SoVITS_V3")

        if ref_audio_path in [None, ""] and (
            (self.prompt_cache["prompt_semantic"] is None) or (self.prompt_cache["refer_spec"] in [None, []])
        ):
            raise ValueError(
                "ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()"
            )

        ###### setting reference audio and prompt text preprocessing ########
        t0 = time.perf_counter()
        if (ref_audio_path is not None) and (ref_audio_path != self.prompt_cache["ref_audio_path"]):
            if not os.path.exists(ref_audio_path):
                raise ValueError(f"{ref_audio_path} not exists")
            self.set_ref_audio(ref_audio_path)

        aux_ref_audio_paths = aux_ref_audio_paths if aux_ref_audio_paths is not None else []
        paths = set(aux_ref_audio_paths) & set(self.prompt_cache["aux_ref_audio_paths"])
        if not (len(list(paths)) == len(aux_ref_audio_paths) == len(self.prompt_cache["aux_ref_audio_paths"])):
            self.prompt_cache["aux_ref_audio_paths"] = aux_ref_audio_paths
            self.prompt_cache["refer_spec"] = [self.prompt_cache["refer_spec"][0]]
            for path in aux_ref_audio_paths:
                if path in [None, ""]:
                    continue
                if not os.path.exists(path):
                    print(i18n("音频文件不存在，跳过："), path)
                    continue
                self.prompt_cache["refer_spec"].append(self._get_ref_spec(path))

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_lang != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
            if self.prompt_cache["prompt_text"] != prompt_text:
                phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                    prompt_text, prompt_lang, self.configs.version
                )
                self.prompt_cache["prompt_text"] = prompt_text
                self.prompt_cache["prompt_lang"] = prompt_lang
                self.prompt_cache["phones"] = phones
                self.prompt_cache["bert_features"] = bert_features
                self.prompt_cache["norm_text"] = norm_text

        ###### text preprocessing ########
        t1 = time.perf_counter()
        data: list = None
        if not return_fragment:
            data = self.text_preprocessor.preprocess(text, text_lang, text_split_method, self.configs.version)
            if len(data) == 0:
                yield 16000, np.zeros(int(16000), dtype=np.int16)
                return

            batch_index_list: list = None
            data, batch_index_list = self.to_batch(
                data,
                prompt_data=self.prompt_cache if not no_prompt_text else None,
                batch_size=batch_size,
                threshold=batch_threshold,
                split_bucket=split_bucket,
                device=self.configs.device,
                precision=self.precision,
            )
        else:
            print(f"############ {i18n('切分文本')} ############")
            texts = self.text_preprocessor.pre_seg_text(text, text_lang, text_split_method)
            data = []
            for i in range(len(texts)):
                if i % batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])

            def make_batch(batch_texts):
                batch_data = []
                print(f"############ {i18n('提取文本Bert特征')} ############")
                for text in tqdm(batch_texts):
                    phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                        text, text_lang, self.configs.version
                    )
                    if phones is None:
                        continue
                    res = {
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                    batch_data.append(res)
                if len(batch_data) == 0:
                    return None
                batch, _ = self.to_batch(
                    batch_data,
                    prompt_data=self.prompt_cache if not no_prompt_text else None,
                    batch_size=batch_size,
                    threshold=batch_threshold,
                    split_bucket=False,
                    device=self.configs.device,
                    precision=self.precision,
                )
                return batch[0]

        t2 = time.perf_counter()
        try:
            print("############ 推理 ############")
            ###### inference ######
            t_34 = 0.0
            t_45 = 0.0
            audio = []
            output_sr = self.configs.sampling_rate if not self.configs.is_v3_synthesizer else 24000
            for item in data:
                t3 = time.perf_counter()
                if return_fragment:
                    item = make_batch(item)
                    if item is None:
                        continue

                batch_phones: List[torch.LongTensor] = item["phones"]
                # batch_phones:torch.LongTensor = item["phones"]
                batch_phones_len: torch.LongTensor = item["phones_len"]
                all_phoneme_ids: torch.LongTensor = item["all_phones"]
                all_phoneme_lens: torch.LongTensor = item["all_phones_len"]
                all_bert_features: torch.LongTensor = item["all_bert_features"]
                norm_text: str = item["norm_text"]
                max_len = item["max_len"]

                print(i18n("前端处理后的文本(每句):"), norm_text)
                if no_prompt_text:
                    prompt = None
                else:
                    prompt = (
                        self.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(self.configs.device)
                    )

                print(f"############ {i18n('预测语义Token')} ############")
                pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_lens,
                    prompt,
                    all_bert_features,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.configs.hz * self.configs.max_sec,
                    max_len=max_len,
                    repetition_penalty=repetition_penalty,
                )
                t4 = time.perf_counter()
                t_34 += t4 - t3

                refer_audio_spec: torch.Tensor = [
                    item.to(dtype=self.precision, device=self.configs.device)
                    for item in self.prompt_cache["refer_spec"]
                ]

                batch_audio_fragment = []

                # ## vits并行推理 method 1
                # pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                # pred_semantic_len = torch.LongTensor([item.shape[0] for item in pred_semantic_list]).to(self.configs.device)
                # pred_semantic = self.batch_sequences(pred_semantic_list, axis=0, pad_value=0).unsqueeze(0)
                # max_len = 0
                # for i in range(0, len(batch_phones)):
                #     max_len = max(max_len, batch_phones[i].shape[-1])
                # batch_phones = self.batch_sequences(batch_phones, axis=0, pad_value=0, max_length=max_len)
                # batch_phones = batch_phones.to(self.configs.device)
                # batch_audio_fragment = (self.vits_model.batched_decode(
                #         pred_semantic, pred_semantic_len, batch_phones, batch_phones_len,refer_audio_spec
                #     ))
                print(f"############ {i18n('合成音频')} ############")
                if not self.configs.is_v3_synthesizer:
                    if speed_factor == 1.0:
                        print(f"{i18n('并行合成中')}...")
                        # ## vits并行推理 method 2
                        pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                        upsample_rate = math.prod(self.vits_model.upsample_rates)
                        audio_frag_idx = [
                            pred_semantic_list[i].shape[0] * 2 * upsample_rate
                            for i in range(0, len(pred_semantic_list))
                        ]
                        audio_frag_end_idx = [sum(audio_frag_idx[: i + 1]) for i in range(0, len(audio_frag_idx))]
                        all_pred_semantic = (
                            torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(self.configs.device)
                        )
                        _batch_phones = torch.cat(batch_phones).unsqueeze(0).to(self.configs.device)
                        _batch_audio_fragment = self.vits_model.decode(
                            all_pred_semantic, _batch_phones, refer_audio_spec, speed=speed_factor
                        ).detach()[0, 0, :]
                        audio_frag_end_idx.insert(0, 0)
                        batch_audio_fragment = [
                            _batch_audio_fragment[audio_frag_end_idx[i - 1] : audio_frag_end_idx[i]]
                            for i in range(1, len(audio_frag_end_idx))
                        ]
                    else:
                        # ## vits串行推理
                        for i, idx in enumerate(tqdm(idx_list)):
                            phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
                            _pred_semantic = (
                                pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0)
                            )  # .unsqueeze(0)#mq要多unsqueeze一次
                            audio_fragment = self.vits_model.decode(
                                _pred_semantic, phones, refer_audio_spec, speed=speed_factor
                            ).detach()[0, 0, :]
                            batch_audio_fragment.append(audio_fragment)  ###试试重建不带上prompt部分
                else:
                    if parallel_infer:
                        print(f"{i18n('并行合成中')}...")
                        audio_fragments = self.v3_synthesis_batched_infer(
                            idx_list, pred_semantic_list, batch_phones, speed=speed_factor, sample_steps=sample_steps
                        )
                        batch_audio_fragment.extend(audio_fragments)
                    else:
                        for i, idx in enumerate(tqdm(idx_list)):
                            phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
                            _pred_semantic = (
                                pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0)
                            )  # .unsqueeze(0)#mq要多unsqueeze一次
                            audio_fragment = self.v3_synthesis(
                                _pred_semantic, phones, speed=speed_factor, sample_steps=sample_steps
                            )
                            batch_audio_fragment.append(audio_fragment)

                t5 = time.perf_counter()
                t_45 += t5 - t4
                if return_fragment:
                    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t4 - t3, t5 - t4))
                    yield self.audio_postprocess(
                        [batch_audio_fragment],
                        output_sr,
                        None,
                        speed_factor,
                        False,
                        fragment_interval,
                        super_sampling if self.configs.is_v3_synthesizer else False,
                    )
                else:
                    audio.append(batch_audio_fragment)

                if self.stop_flag:
                    yield 16000, np.zeros(int(16000), dtype=np.int16)
                    return

            if not return_fragment:
                print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t_34, t_45))
                if len(audio) == 0:
                    yield 16000, np.zeros(int(16000), dtype=np.int16)
                    return
                yield self.audio_postprocess(
                    audio,
                    output_sr,
                    batch_index_list,
                    speed_factor,
                    split_bucket,
                    fragment_interval,
                    super_sampling if self.configs.is_v3_synthesizer else False,
                )

        except Exception as e:
            traceback.print_exc()
            # 必须返回一个空音频, 否则会导致显存不释放。
            yield 16000, np.zeros(int(16000), dtype=np.int16)
            # 重置模型, 否则会导致显存释放不完全。
            del self.t2s_model
            del self.vits_model
            self.t2s_model = None
            self.vits_model = None
            self.init_t2s_weights(self.configs.t2s_weights_path)
            self.init_vits_weights(self.configs.vits_weights_path)
            raise e
        finally:
            self.empty_cache()

    def empty_cache(self):
        try:
            gc.collect()  # 触发gc的垃圾回收。避免内存一直增长。
            if "cuda" in str(self.configs.device):
                torch.cuda.empty_cache()
            elif str(self.configs.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass

    def audio_postprocess(
        self,
        audio: List[torch.Tensor],
        sr: int,
        batch_index_list: list = None,
        speed_factor: float = 1.0,
        split_bucket: bool = True,
        fragment_interval: float = 0.3,
        super_sampling: bool = False,
    ) -> Tuple[int, np.ndarray]:
        zero_wav = torch.zeros(
            int(self.configs.sampling_rate * fragment_interval), dtype=self.precision, device=self.configs.device
        )

        for i, batch in enumerate(audio):
            for j, audio_fragment in enumerate(batch):
                max_audio = torch.abs(audio_fragment).max()  # 简单防止16bit爆音
                if max_audio > 1:
                    audio_fragment /= max_audio
                audio_fragment: torch.Tensor = torch.cat([audio_fragment, zero_wav], dim=0)
                audio[i][j] = audio_fragment

        if split_bucket:
            audio = self.recovery_order(audio, batch_index_list)
        else:
            # audio = [item for batch in audio for item in batch]
            audio = sum(audio, [])

        audio = torch.cat(audio, dim=0)

        if super_sampling:
            print(f"############ {i18n('音频超采样')} ############")
            t1 = time.perf_counter()
            self.init_sr_model()
            if not self.sr_model_not_exist:
                audio, sr = self.sr_model(audio.unsqueeze(0), sr)
                max_audio = np.abs(audio).max()
                if max_audio > 1:
                    audio /= max_audio
            t2 = time.perf_counter()
            print(f"超采样用时：{t2 - t1:.3f}s")
        else:
            audio = audio.cpu().numpy()

        audio = (audio * 32768).astype(np.int16)

        # try:
        #     if speed_factor != 1.0:
        #         audio = speed_change(audio, speed=speed_factor, sr=int(sr))
        # except Exception as e:
        #     print(f"Failed to change speed of audio: \n{e}")

        return sr, audio

    def v3_synthesis(
        self, semantic_tokens: torch.Tensor, phones: torch.Tensor, speed: float = 1.0, sample_steps: int = 32
    ):
        prompt_semantic_tokens = self.prompt_cache["prompt_semantic"].unsqueeze(0).unsqueeze(0).to(self.configs.device)
        prompt_phones = torch.LongTensor(self.prompt_cache["phones"]).unsqueeze(0).to(self.configs.device)
        refer_audio_spec = self.prompt_cache["refer_spec"][0].to(dtype=self.precision, device=self.configs.device)

        fea_ref, ge = self.vits_model.decode_encp(prompt_semantic_tokens, prompt_phones, refer_audio_spec)
        ref_audio: torch.Tensor = self.prompt_cache["raw_audio"]
        ref_sr = self.prompt_cache["raw_sr"]
        ref_audio = ref_audio.to(self.configs.device).float()
        if ref_audio.shape[0] == 2:
            ref_audio = ref_audio.mean(0).unsqueeze(0)
        if ref_sr != 24000:
            ref_audio = resample(ref_audio, ref_sr, self.configs.device)

        mel2 = mel_fn(ref_audio)
        mel2 = norm_spec(mel2)
        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        if T_min > 468:
            mel2 = mel2[:, :, -468:]
            fea_ref = fea_ref[:, :, -468:]
            T_min = 468
        chunk_len = 934 - T_min

        mel2 = mel2.to(self.precision)
        fea_todo, ge = self.vits_model.decode_encp(semantic_tokens, phones, refer_audio_spec, ge, speed)

        cfm_resss = []
        idx = 0
        while 1:
            fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break
            idx += chunk_len
            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)

            cfm_res = self.vits_model.cfm.inference(
                fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
            )
            cfm_res = cfm_res[:, :, mel2.shape[2] :]

            mel2 = cfm_res[:, :, -T_min:]
            fea_ref = fea_todo_chunk[:, :, -T_min:]

            cfm_resss.append(cfm_res)
        cfm_res = torch.cat(cfm_resss, 2)
        cfm_res = denorm_spec(cfm_res)

        with torch.inference_mode():
            wav_gen = self.bigvgan_model(cfm_res)
            audio = wav_gen[0][0]  # .cpu().detach().numpy()

        return audio

    def v3_synthesis_batched_infer(
        self,
        idx_list: List[int],
        semantic_tokens_list: List[torch.Tensor],
        batch_phones: List[torch.Tensor],
        speed: float = 1.0,
        sample_steps: int = 32,
    ) -> List[torch.Tensor]:
        prompt_semantic_tokens = self.prompt_cache["prompt_semantic"].unsqueeze(0).unsqueeze(0).to(self.configs.device)
        prompt_phones = torch.LongTensor(self.prompt_cache["phones"]).unsqueeze(0).to(self.configs.device)
        refer_audio_spec = self.prompt_cache["refer_spec"][0].to(dtype=self.precision, device=self.configs.device)

        fea_ref, ge = self.vits_model.decode_encp(prompt_semantic_tokens, prompt_phones, refer_audio_spec)
        ref_audio: torch.Tensor = self.prompt_cache["raw_audio"]
        ref_sr = self.prompt_cache["raw_sr"]
        ref_audio = ref_audio.to(self.configs.device).float()
        if ref_audio.shape[0] == 2:
            ref_audio = ref_audio.mean(0).unsqueeze(0)
        if ref_sr != 24000:
            ref_audio = resample(ref_audio, ref_sr, self.configs.device)

        mel2 = mel_fn(ref_audio)
        mel2 = norm_spec(mel2)
        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        if T_min > 468:
            mel2 = mel2[:, :, -468:]
            fea_ref = fea_ref[:, :, -468:]
            T_min = 468
        chunk_len = 934 - T_min

        mel2 = mel2.to(self.precision)

        # #### batched inference
        overlapped_len = 12
        feat_chunks = []
        feat_lens = []
        feat_list = []

        for i, idx in enumerate(idx_list):
            phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
            semantic_tokens = (
                semantic_tokens_list[i][-idx:].unsqueeze(0).unsqueeze(0)
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            feat, _ = self.vits_model.decode_encp(semantic_tokens, phones, refer_audio_spec, ge, speed)
            feat_list.append(feat)
            feat_lens.append(feat.shape[2])

        feats = torch.cat(feat_list, 2)
        feats_padded = F.pad(feats, (overlapped_len, 0), "constant", 0)
        pos = 0
        padding_len = 0
        while True:
            if pos == 0:
                chunk = feats_padded[:, :, pos : pos + chunk_len]
            else:
                pos = pos - overlapped_len
                chunk = feats_padded[:, :, pos : pos + chunk_len]
            pos += chunk_len
            if chunk.shape[-1] == 0:
                break

            # padding for the last chunk
            padding_len = chunk_len - chunk.shape[2]
            if padding_len != 0:
                chunk = F.pad(chunk, (0, padding_len), "constant", 0)
            feat_chunks.append(chunk)

        feat_chunks = torch.cat(feat_chunks, 0)
        bs = feat_chunks.shape[0]
        fea_ref = fea_ref.repeat(bs, 1, 1)
        fea = torch.cat([fea_ref, feat_chunks], 2).transpose(2, 1)
        pred_spec = self.vits_model.cfm.inference(
            fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
        )
        pred_spec = pred_spec[:, :, -chunk_len:]
        dd = pred_spec.shape[1]
        pred_spec = pred_spec.permute(1, 0, 2).contiguous().view(dd, -1).unsqueeze(0)
        # pred_spec = pred_spec[..., :-padding_len]

        pred_spec = denorm_spec(pred_spec)

        with torch.no_grad():
            wav_gen = self.bigvgan_model(pred_spec)
            audio = wav_gen[0][0]  # .cpu().detach().numpy()

        audio_fragments = []
        upsample_rate = 256
        pos = 0

        while pos < audio.shape[-1]:
            audio_fragment = audio[pos : pos + chunk_len * upsample_rate]
            audio_fragments.append(audio_fragment)
            pos += chunk_len * upsample_rate

        audio = self.sola_algorithm(audio_fragments, overlapped_len * upsample_rate)
        audio = audio[overlapped_len * upsample_rate : -padding_len * upsample_rate]

        audio_fragments = []
        for feat_len in feat_lens:
            audio_fragment = audio[: feat_len * upsample_rate]
            audio_fragments.append(audio_fragment)
            audio = audio[feat_len * upsample_rate :]

        return audio_fragments

    def sola_algorithm(
        self,
        audio_fragments: List[torch.Tensor],
        overlap_len: int,
    ):
        for i in range(len(audio_fragments) - 1):
            f1 = audio_fragments[i]
            f2 = audio_fragments[i + 1]
            w1 = f1[-overlap_len:]
            w2 = f2[:overlap_len]
            assert w1.shape == w2.shape
            corr = F.conv1d(w1.view(1, 1, -1), w2.view(1, 1, -1), padding=w2.shape[-1] // 2).view(-1)[:-1]
            idx = corr.argmax()
            f1_ = f1[: -(overlap_len - idx)]
            audio_fragments[i] = f1_

            f2_ = f2[idx:]
            window = torch.hann_window((overlap_len - idx) * 2, device=f1.device, dtype=f1.dtype)
            f2_[: (overlap_len - idx)] = (
                window[: (overlap_len - idx)] * f2_[: (overlap_len - idx)]
                + window[(overlap_len - idx) :] * f1[-(overlap_len - idx) :]
            )
            audio_fragments[i + 1] = f2_

        return torch.cat(audio_fragments, 0)
