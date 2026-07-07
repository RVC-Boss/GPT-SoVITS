from __future__ import annotations

"""
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
"""
import psutil
import os

def set_high_priority():
    """把当前 Python 进程设为 HIGH_PRIORITY_CLASS"""
    if os.name != "nt":
        return # 仅 Windows 有效
    p = psutil.Process(os.getpid())
    try:
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("已将进程优先级设为 High")
    except psutil.AccessDenied:
        print("权限不足，无法修改优先级（请用管理员运行）")
set_high_priority()
import json
import logging
import os
import re
import sys
import traceback
import warnings
from pathlib import Path

import torch
import torchaudio
from text.LangSegmenter import LangSegmenter

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)

version = model_version = os.environ.get("version", "v2")

from config import change_choices, get_weights_names, name2gpt_path, name2sovits_path

SoVITS_names, GPT_names = get_weights_names()
from config import pretrained_sovits_name

path_sovits_v3 = pretrained_sovits_name["v3"]
path_sovits_v4 = pretrained_sovits_name["v4"]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)

if os.path.exists("./weight.json"):
    pass
else:
    with open("./weight.json", "w", encoding="utf-8") as file:
        json.dump({"GPT": {}, "SoVITS": {}}, file)

with open("./weight.json", "r", encoding="utf-8") as file:
    weight_data = file.read()
    weight_data = json.loads(weight_data)
    gpt_path = os.environ.get("gpt_path", weight_data.get("GPT", {}).get(version, GPT_names[-1]))
    sovits_path = os.environ.get("sovits_path", weight_data.get("SoVITS", {}).get(version, SoVITS_names[0]))
    if isinstance(gpt_path, list):
        gpt_path = gpt_path[0]
    if isinstance(sovits_path, list):
        sovits_path = sovits_path[0]

# print(2333333)
# print(os.environ["gpt_path"])
# print(gpt_path)
# print(GPT_names)
# print(weight_data)
# print(weight_data.get("GPT", {}))
# print(version)###GPT version里没有s2的v2pro
# print(weight_data.get("GPT", {}).get(version, GPT_names[-1]))

cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
# is_half=False
punctuation = set(["!", "?", "…", ",", ".", "-", " "])
import gradio as gr
import librosa
import numpy as np
from feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer

cnhubert.cnhubert_base_path = cnhubert_base_path

import random

from GPT_SoVITS.module.models import Generator, SynthesizerTrn, SynthesizerTrnV3


def set_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000000)
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# set_seed(42)

from time import time as ttime

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from peft import LoraConfig, get_peft_model
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def check_cuda_graph_support():
    if device != "cuda":
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 7:
            print("CUDA Graph: GPU compute capability < 7.0, disabled")
            return False
        a = torch.randn(2, 2, device="cuda")
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            b = a * 2
        torch.cuda.current_stream().wait_stream(s)
        out = torch.empty_like(b)
        with torch.cuda.graph(g):
            out.copy_(a * 2)
        g.replay()
        torch.cuda.synchronize()
        del a, b, out, g, s
        torch.cuda.empty_cache()
        print("CUDA Graph: support check passed, auto-enabled")
        return True
    except Exception as e:
        print(f"CUDA Graph: support check failed ({e}), disabled")
        return False


cuda_graph_supported = check_cuda_graph_support()

dict_language_v1 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("粤语"): "all_yue",  # 全部按中文识别
    i18n("韩文"): "all_ko",  # 全部按韩文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("粤英混合"): "yue",  # 按粤英混合识别####不变
    i18n("韩英混合"): "ko",  # 按韩英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种
}
dict_language = dict_language_v1 if version == "v1" else dict_language_v2

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


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


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


###todo:put them to process_ckpt and modify my_save func (save sovits weights), gpt save weights use my_save in process_ckpt
# symbol_version-model_version-if_lora_v3
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

v3v4set = {"v3", "v4"}


def change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
    if "！" in sovits_path or "!" in sovits_path:
        sovits_path = name2sovits_path[sovits_path]
    global vq_model, hps, version, model_version, dict_language, if_lora_v3, t2s_model_cudagraph
    t2s_model_cudagraph = None
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    print(sovits_path, version, model_version, if_lora_v3)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    if if_lora_v3 == True and is_exist == False:
        info = path_sovits + "SoVITS %s" % model_version + i18n("底模缺失，无法加载相应 LoRA 权重")
        gr.Warning(info)
        raise FileExistsError(info)
    dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = (
                {"__type__": "update"},
                {"__type__": "update", "value": prompt_language},
            )
        else:
            prompt_text_update = {"__type__": "update", "value": ""}
            prompt_language_update = {"__type__": "update", "value": i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {"__type__": "update"}, {"__type__": "update", "value": text_language}
        else:
            text_update = {"__type__": "update", "value": ""}
            text_language_update = {"__type__": "update", "value": i18n("中文")}
        if model_version in v3v4set:
            visible_sample_steps = True
            visible_inp_refs = False
        else:
            visible_sample_steps = False
            visible_inp_refs = True
        yield (
            {"__type__": "update", "choices": list(dict_language.keys())},
            {"__type__": "update", "choices": list(dict_language.keys())},
            prompt_text_update,
            prompt_language_update,
            text_update,
            text_language_update,
            {
                "__type__": "update",
                "visible": visible_sample_steps,
                "value": 32 if model_version == "v3" else 8,
                "choices": [4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
            },
            {"__type__": "update", "visible": visible_inp_refs},
            {"__type__": "update", "value": False, "interactive": True if model_version not in v3v4set else False},
            {"__type__": "update", "visible": True if model_version == "v3" else False},
            {"__type__": "update", "value": i18n("模型加载中，请等待"), "interactive": False},
        )

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"  # v3model,v2sybomls
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    # print("sovits版本:",hps.model.version)
    if model_version not in v3v4set:
        if "Pro" not in model_version:
            model_version = version
        else:
            hps.model.version = model_version
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    if if_lora_v3 == False:
        print("loading sovits_%s" % model_version, vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else:
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        print(
            "loading sovits_%spretrained_G" % model_version,
            vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False),
        )
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print("loading sovits_%s_lora%s" % (model_version, lora_rank))
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        # torch.save(vq_model.state_dict(),"merge_win.pth")
        vq_model.eval()

    yield (
        {"__type__": "update", "choices": list(dict_language.keys())},
        {"__type__": "update", "choices": list(dict_language.keys())},
        prompt_text_update,
        prompt_language_update,
        text_update,
        text_language_update,
        {
            "__type__": "update",
            "visible": visible_sample_steps,
            "value": 32 if model_version == "v3" else 8,
            "choices": [4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
        },
        {"__type__": "update", "visible": visible_inp_refs},
        {"__type__": "update", "value": False, "interactive": True if model_version not in v3v4set else False},
        {"__type__": "update", "visible": True if model_version == "v3" else False},
        {"__type__": "update", "value": i18n("合成语音"), "interactive": True},
    )
    with open("./weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as f:
        f.write(json.dumps(data))


try:
    next(change_sovits_weights(sovits_path))
except:
    pass


t2s_model_cudagraph = None
gpt_path_global = None


def change_gpt_weights(gpt_path):
    if "！" in gpt_path or "!" in gpt_path:
        gpt_path = name2gpt_path[gpt_path]
    global hz, max_sec, t2s_model, config, t2s_model_cudagraph, gpt_path_global
    t2s_model_cudagraph = None
    gpt_path_global = gpt_path
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    # total = sum([param.nelement() for param in t2s_model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["GPT"][version] = gpt_path
    with open("./weight.json", "w") as f:
        f.write(json.dumps(data))


change_gpt_weights(gpt_path)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch

now_dir = os.getcwd()


def clean_hifigan_model():
    global hifigan_model
    if hifigan_model:
        hifigan_model = hifigan_model.cpu()
        hifigan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def clean_bigvgan_model():
    global bigvgan_model
    if bigvgan_model:
        bigvgan_model = bigvgan_model.cpu()
        bigvgan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def clean_sv_cn_model():
    global sv_cn_model
    if sv_cn_model:
        sv_cn_model.embedding_model = sv_cn_model.embedding_model.cpu()
        sv_cn_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def init_bigvgan():
    global bigvgan_model, hifigan_model, sv_cn_model
    from BigVGAN import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False,
    )  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    clean_hifigan_model()
    clean_sv_cn_model()
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)


def init_hifigan():
    global hifigan_model, bigvgan_model, sv_cn_model
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0,
        is_bias=True,
    )
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load(
        "%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,),
        map_location="cpu",
        weights_only=False,
    )
    print("loading vocoder", hifigan_model.load_state_dict(state_dict_g))
    clean_bigvgan_model()
    clean_sv_cn_model()
    if is_half == True:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)


from sv import SV


def init_sv_cn():
    global hifigan_model, bigvgan_model, sv_cn_model
    sv_cn_model = SV(device, is_half)
    clean_bigvgan_model()
    clean_hifigan_model()


bigvgan_model = hifigan_model = sv_cn_model = None
if model_version == "v3":
    init_bigvgan()
if model_version == "v4":
    init_hifigan()
if model_version in {"v2Pro", "v2ProPlus"}:
    init_sv_cn()

resample_transform_dict = {}


def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    # audio = load_audio(filename, int(hps.data.sampling_rate))

    # audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
    # audio = torch.FloatTensor(audio)

    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, device)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro == True:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio


def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


dtype = torch.float16 if is_half == True else torch.float32


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


from text import chinese


def get_phones_and_bert(text, language, version, final=False):
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text,"ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text,"ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(dtype), norm_text


from module.mel_processing import mel_spectrogram_torch, spectrogram_torch

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
mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


sr_model = None


def audio_sr(audio, sr):
    global sr_model
    if sr_model == None:
        from tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(device, DictToAttrRecursive)
        except FileNotFoundError:
            gr.Warning(i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)


##ref_wav_path+prompt_text+prompt_language+text(单个)+text_language+top_k+top_p+temperature
# cache_tokens={}#暂未实现清理机制
cache = {}


def get_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut=i18n("不切"),
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free=False,
    speed=1,
    if_freeze=False,
    inp_refs=None,
    sample_steps=8,
    if_sr=False,
    pause_second=0.3,
    use_cuda_graph=False,
):
    global cache
    if ref_wav_path:
        pass
    else:
        gr.Warning(i18n("请上传参考音频"))
    if text:
        pass
    else:
        gr.Warning(i18n("请填入推理文本"))
    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    if model_version in v3v4set:
        ref_free = False  # s2v3暂不支持ref_free
    else:
        if_sr = False
    if model_version not in {"v3", "v4", "v2Pro", "v2ProPlus"}:
        clean_bigvgan_model()
        clean_hifigan_model()
        clean_sv_cn_model()
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    text = text.strip("\n")
    # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half == True else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half == True:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            ref_duration = wav16k.shape[0] / 16000
            if ref_duration < 3 or ref_duration > 10:
                gr.Warning(i18n(f"参考音频时长 {ref_duration:.1f} 秒，超出推荐的 3~10 秒范围，可能影响合成质量"))
            wav16k = torch.from_numpy(wav16k)
            if is_half == True:
                wav16k = wav16k.half().to(device)
            else:
                wav16k = wav16k.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1 - t0)

    if how_to_cut == i18n("凑四句一切"):
        text = cut1(text)
    elif how_to_cut == i18n("凑50字一切"):
        text = cut2(text)
    elif how_to_cut == i18n("按中文句号。切"):
        text = cut3(text)
    elif how_to_cut == i18n("按英文句号.切"):
        text = cut4(text)
    elif how_to_cut == i18n("按标点符号切"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    ###s2v3暂不支持ref_free
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)

    for i_text, text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
        print(i18n("前端处理后的文本(每句):"), norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
        # print(cache.keys(),if_freeze)
        if i_text in cache and if_freeze == True:
            pred_semantic = cache[i_text]
        else:
            if use_cuda_graph and device == "cuda":
                global t2s_model_cudagraph
                if t2s_model_cudagraph is None:
                    from AR.models.t2s_model_cudagraph import CUDAGraphRunner
                    t2s_model_cudagraph = CUDAGraphRunner(
                        CUDAGraphRunner.load_decoder(gpt_path_global),
                        torch.device(device),
                        torch.float16 if is_half else torch.float32,
                    )
                from AR.models.structs_cudagraph import T2SRequest
                with torch.no_grad():
                    t2s_request = T2SRequest(
                        [all_phoneme_ids.squeeze(0)],
                        all_phoneme_len,
                        all_phoneme_ids.new_zeros((1, 0)) if ref_free else prompt,
                        [bert.squeeze(0)],
                        valid_length=1,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=hz * max_sec,
                        use_cuda_graph=True,
                    )
                    t2s_result = t2s_model_cudagraph.generate(t2s_request)
                    if t2s_result.exception is not None:
                        print(t2s_result.exception)
                        print(t2s_result.traceback)
                        raise RuntimeError("CUDA Graph T2S inference failed")
                    pred_semantic = t2s_result.result[0].unsqueeze(0).unsqueeze(0)
                    cache[i_text] = pred_semantic
            else:
                with torch.no_grad():
                    pred_semantic, idx = t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_len,
                        None if ref_free else prompt,
                        bert,
                        # prompt_phone_len=ph_offset,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=hz * max_sec,
                    )
                    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                    cache[i_text] = pred_semantic
        t3 = ttime()
        is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
        # print(23333,is_v2pro,model_version)
        ###v3不存在以下逻辑和inp_refs
        if model_version not in v3v4set:
            refers = []
            if is_v2pro:
                sv_emb = []
                if sv_cn_model == None:
                    init_sv_cn()
            if inp_refs:
                for path in inp_refs:
                    try:  #####这里加上提取sv的逻辑，要么一堆sv一堆refer，要么单个sv单个refer
                        refer, audio_tensor = get_spepc(hps, path.name, dtype, device, is_v2pro)
                        refers.append(refer)
                        if is_v2pro:
                            sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor))
                    except:
                        traceback.print_exc()
            if len(refers) == 0:
                refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device, is_v2pro)
                refers = [refers]
                if is_v2pro:
                    sv_emb = [sv_cn_model.compute_embedding3(audio_tensor)]
            if is_v2pro:
                audio = vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed, sv_emb=sv_emb
                )[0][0]
            else:
                audio = vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed
                )[0][0]
        else:
            refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
            phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
            phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
            fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
            ref_audio, sr = torchaudio.load(ref_wav_path)
            ref_audio = ref_audio.to(device).float()
            if ref_audio.shape[0] == 2:
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            tgt_sr = 24000 if model_version == "v3" else 32000
            if sr != tgt_sr:
                ref_audio = resample(ref_audio, sr, tgt_sr, device)
            # print("ref_audio",ref_audio.abs().mean())
            mel2 = mel_fn(ref_audio) if model_version == "v3" else mel_fn_v4(ref_audio)
            mel2 = norm_spec(mel2)
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            Tref = 468 if model_version == "v3" else 500
            Tchunk = 934 if model_version == "v3" else 1000
            if T_min > Tref:
                mel2 = mel2[:, :, -Tref:]
                fea_ref = fea_ref[:, :, -Tref:]
                T_min = Tref
            chunk_len = Tchunk - T_min
            mel2 = mel2.to(dtype)
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
            cfm_resss = []
            idx = 0
            while 1:
                fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
                if fea_todo_chunk.shape[-1] == 0:
                    break
                idx += chunk_len
                fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                cfm_res = vq_model.cfm.inference(
                    fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
                )
                cfm_res = cfm_res[:, :, mel2.shape[2] :]
                mel2 = cfm_res[:, :, -T_min:]
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                cfm_resss.append(cfm_res)
            cfm_res = torch.cat(cfm_resss, 2)
            cfm_res = denorm_spec(cfm_res)
            if model_version == "v3":
                if bigvgan_model == None:
                    init_bigvgan()
            else:  # v4
                if hifigan_model == None:
                    init_hifigan()
            vocoder_model = bigvgan_model if model_version == "v3" else hifigan_model
            with torch.inference_mode():
                wav_gen = vocoder_model(cfm_res)
                audio = wav_gen[0][0]  # .cpu().detach().numpy()
        max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio = audio / max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)  # zero_wav
        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
    audio_opt = torch.cat(audio_opt, 0)  # np.concatenate
    if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000  # v4
    if if_sr == True and opt_sr == 24000:
        print(i18n("音频超分中"))
        audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
        max_audio = np.abs(audio_opt).max()
        if max_audio > 1:
            audio_opt /= max_audio
    else:
        audio_opt = audio_opt.cpu().detach().numpy()
    yield opt_sr, (audio_opt * 32767).astype(np.int16)


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split("(\d+)", s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def html_center(text, label="p"):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


def html_left(text, label="p"):
    return f"""<div style="text-align: left; margin: 0; padding: 0;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


# ===================== 训练角色选择 / 训练样本浏览 辅助函数 =====================
def scan_model_names() -> list[str]:
    """扫描 logs/ 目录获取所有训练角色名（含 2-name2text.txt 的子目录）。"""
    from config import exp_root
    import os

    root = Path(exp_root)
    if not root.exists():
        return []
    return sorted(
        [
            d.name
            for d in root.iterdir()
            if d.is_dir() and (d / "2-name2text.txt").exists()
        ]
    )


def refresh_model_dropdown():
    """刷新模型下拉框：返回 gr.Dropdown 更新（choices=全部模型名, value=首个）。

    供 app.load 和「刷新模型列表」按钮使用。
    注意：必须返回 gr.Dropdown(...) 而非裸 list——裸 list 会被 Gradio 当作
    dropdown 的 value（多选），导致级联的 on_model_name_change 收到 list 而非 str。
    """
    names = scan_model_names()
    return gr.Dropdown(choices=names, value=names[0] if names else "")


def _coerce_single(value):
    """把可能被 Gradio 包装成 list/tuple 的单值还原为 str。"""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value) if value is not None else ""



def _read_name2text_for_webui(logs_dir: Path) -> dict[str, dict[str, str]]:
    """读取 2-name2text.txt（WebUI 版，与 api_v2.py 的 _read_name2text 逻辑一致）。

    返回 {wav_name: {"text": ..., "lang": ...}}，同时以 stem 建索引。
    """
    path = logs_dir / "2-name2text.txt"
    if not path.exists():
        return {}
    output: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        wav_name = parts[0].strip()
        text = parts[3].strip()
        if wav_name and text:
            entry = {"text": text, "lang": "zh"}
            output[wav_name] = entry
            output[Path(wav_name).stem] = entry
    return output


def _read_emotion_map_for_webui(model_name: str) -> dict[str, str]:
    """读取 ASR 标注 .list 中的 emotion 列，返回 {audio_name|stem: emotion}。

    扫描 output/asr_opt/<model_name>/ 下所有 *.list（4/5/6 列均兼容）。
    无 emotion 列或文件不存在时返回空 dict。
    """
    from tools.list_metadata import parse_list_line

    emotion_map: dict[str, str] = {}
    list_dir = Path("output") / "asr_opt" / model_name
    if not list_dir.exists():
        return emotion_map
    for list_path in sorted(list_dir.glob("*.list")):
        try:
            for line in list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                item = parse_list_line(line)
                if item is None or not item.emotion:
                    continue
                name = os.path.basename(item.wav_path.strip("\"'"))
                if not name:
                    continue
                emotion_map[name] = item.emotion
                emotion_map[Path(name).stem] = item.emotion
        except Exception:
            continue
    return emotion_map


def on_model_name_change(model_name: str):
    """模型名变化时：加载训练样本列表 + 自动匹配最佳 GPT/SoVITS 权重。

    返回: [样本下拉框, GPT下拉框, SoVITS下拉框]
    样本下拉框 value 设为首个样本, 会级联触发 on_ref_sample_change 自动引用为参考音频。
    GPT/SoVITS 自动选中 epoch 最接近推荐值(8/15)的权重, 用户仍可手动改。
    """
    model_name = _coerce_single(model_name)
    if not model_name:
        return gr.Dropdown(choices=[], value=""), gr.Dropdown(), gr.Dropdown()
    from config import exp_root

    # 1. 加载训练样本
    logs_dir = Path(exp_root) / model_name
    wav_dir = logs_dir / "5-wav32k"
    samples: list[tuple[str, str]] = []
    if wav_dir.exists():
        name2text = _read_name2text_for_webui(logs_dir)
        emotion_map = _read_emotion_map_for_webui(model_name)
        for f in sorted(wav_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".wav", ".mp3", ".flac"):
                text = name2text.get(f.name, {}).get("text", "") or name2text.get(f.stem, {}).get("text", "")
                emotion = emotion_map.get(f.name, "") or emotion_map.get(f.stem, "")
                label = f"{f.name}"
                if text:
                    label += f" | {text[:30]}"
                if emotion:
                    label += f" | 【{emotion}】"
                samples.append((label, str(f)))
    choices = [s[0] for s in samples]
    value = samples[0][0] if samples else ""
    # 2. 自动匹配 GPT/SoVITS 权重（epoch 最接近 8/15）
    gpt_dd, sovits_dd = auto_match_weights_for_model(model_name)
    return gr.Dropdown(choices=choices, value=value), gpt_dd, sovits_dd


def on_ref_sample_change(sample_label: str, model_name: str):
    """训练样本选择变化时，直接引用为参考音频 + 填充参考文本 + 情绪/括注。

    切换「训练样本音频」下拉框即自动联动, 无需额外按钮。
    用户手动清空参考音频/文本/情绪则视为自定义输入, 不被本函数干扰
    （Gradio change 仅在 dropdown 变化时触发, 手动改文本框不会重置）。

    返回: [参考音频(inp_ref), 参考文本(prompt_text), 情绪/括注(ref_emotion_text)]
    """
    sample_label = _coerce_single(sample_label)
    model_name = _coerce_single(model_name)
    if not sample_label or not model_name:
        return None, "", ""
    from config import exp_root

    wav_name = sample_label.split(" | ")[0]
    audio_path = Path(exp_root) / model_name / "5-wav32k" / wav_name
    logs_dir = Path(exp_root) / model_name
    name2text = _read_name2text_for_webui(logs_dir)
    emotion_map = _read_emotion_map_for_webui(model_name)
    text = name2text.get(wav_name, {}).get("text", "") or name2text.get(Path(wav_name).stem, {}).get("text", "")
    emotion = emotion_map.get(wav_name, "") or emotion_map.get(Path(wav_name).stem, "")
    audio_out = str(audio_path) if audio_path.exists() else None
    return audio_out, text, emotion


def _scan_model_weights_for_webui() -> dict[str, dict[str, list[str]]]:
    """扫描权重目录（WebUI 版），按 exp_name 分组。"""
    from config import GPT_weight_root, SoVITS_weight_root

    grouped: dict[str, dict[str, list[str]]] = {}
    for root in GPT_weight_root:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for f in root_path.iterdir():
            if f.is_file() and f.suffix == ".ckpt":
                name = re.split(r"-e\d+", f.stem, flags=re.IGNORECASE)[0]
                grouped.setdefault(name, {"gpt": [], "sovits": []})
                grouped[name]["gpt"].append(f"{root}/{f.name}")
    for root in SoVITS_weight_root:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for f in root_path.iterdir():
            if f.is_file() and f.suffix == ".pth":
                name = re.split(r"_e\d+_s\d+", f.stem, flags=re.IGNORECASE)[0]
                grouped.setdefault(name, {"gpt": [], "sovits": []})
                grouped[name]["sovits"].append(f"{root}/{f.name}")
    return grouped


def _extract_epoch_from_weight(weight_path: str, kind: str) -> int | None:
    """从权重文件名提取 epoch 数字。

    kind='gpt':    '<exp>-e10.ckpt'    -> 10
    kind='sovits': '<exp>_e12_s180.pth' -> 12
    """
    name = Path(weight_path).name
    pattern = r"-e(\d+)" if kind == "gpt" else r"_e(\d+)_s\d+"
    m = re.search(pattern, name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _pick_weight_by_epoch(weights: list[str], target_epoch: int) -> str | None:
    """从权重列表中选 epoch 最接近 target_epoch 的（平手取较小 epoch）。

    匹配训练 WebUI 的推荐: GPT 目标 8 (total_epoch 默认 8), SoVITS 目标 15
    (total_epoch 默认 15)。无法提取 epoch 时退化为列表第一个。
    """
    if not weights:
        return None
    best = None
    best_diff = None
    for w in weights:
        # kind 由文件扩展名推断
        kind = "gpt" if w.endswith(".ckpt") else "sovits"
        ep = _extract_epoch_from_weight(w, kind)
        if ep is None:
            continue
        diff = abs(ep - target_epoch)
        if best_diff is None or diff < best_diff or (diff == best_diff and ep < (best_ep or 0)):
            best = w
            best_diff = diff
            best_ep = ep
    return best or weights[0]


# GPT/SoVITS 自动匹配权重的目标 epoch（对应训练 WebUI 的默认 total_epoch）
_GPT_TARGET_EPOCH = 8
_SOVITS_TARGET_EPOCH = 15


def auto_match_weights_for_model(model_name: str):
    """选中模型名时自动匹配最佳 GPT/SoVITS 权重（按目标 epoch 最近匹配）。

    返回 (gpt_dropdown_update, sovits_dropdown_update)；真正的权重切换由已绑定的
    GPT_dropdown.change / SoVITS_dropdown.change 在 value 变化时自动触发。
    """
    if not model_name:
        return gr.Dropdown(), gr.Dropdown()
    weights = _scan_model_weights_for_webui()
    model_weights = weights.get(model_name, {"gpt": [], "sovits": []})
    gpt_best = _pick_weight_by_epoch(model_weights["gpt"], _GPT_TARGET_EPOCH)
    sovits_best = _pick_weight_by_epoch(model_weights["sovits"], _SOVITS_TARGET_EPOCH)
    return gr.Dropdown(value=gpt_best), gr.Dropdown(value=sovits_best)


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:
    gr.HTML(
        top_html.format(
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.")
            + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        ),
        elem_classes="markdown",
    )
    with gr.Group():
        gr.Markdown(html_center(i18n("模型切换"), "h3"))
        # 模型名(实验名) 是 GPT/SoVITS 的上游选择: 选角色 -> 自动匹配权重
        with gr.Row():
            model_name_dropdown = gr.Dropdown(
                label=i18n("模型名（训练实验名）"),
                choices=[],
                value="",
                interactive=True,
                scale=21,
            )
            refresh_models_btn = gr.Button(i18n("刷新模型列表"), variant="primary", scale=7)
        with gr.Row():
            GPT_dropdown = gr.Dropdown(
                label=i18n("GPT模型列表"),
                choices=sorted(GPT_names, key=custom_sort_key),
                value=gpt_path,
                interactive=True,
                scale=14,
            )
            SoVITS_dropdown = gr.Dropdown(
                label=i18n("SoVITS模型列表"),
                choices=sorted(SoVITS_names, key=custom_sort_key),
                value=sovits_path,
                interactive=True,
                scale=14,
            )
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary", scale=14)
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
    with gr.Group():
        gr.Markdown(html_center(i18n("*请上传并填写参考信息"), "h3"))
        # 训练样本下拉: 入口与输出聚合 — 选中即自动引用为参考音频+文本+情绪
        ref_sample_dropdown = gr.Dropdown(
            label=i18n("训练样本音频（选中即自动引用为参考音频）"),
            choices=[],
            value="",
            interactive=True,
        )
        with gr.Row():
            inp_ref = gr.Audio(label=i18n("请上传参考音频（推荐3~10秒）"), type="filepath", scale=13)
            with gr.Column(scale=13):
                ref_text_free = gr.Checkbox(
                    label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。")
                    + i18n("v3暂不支持该模式，使用了会报错。"),
                    value=False,
                    interactive=True if model_version not in v3v4set else False,
                    show_label=True,
                    scale=1,
                )
                gr.Markdown(
                    html_left(
                        i18n("使用无参考文本模式时建议使用微调的GPT")
                        + "<br>"
                        + i18n("听不清参考音频说的啥(不晓得写啥)可以开。开启后无视填写的参考文本。")
                    )
                )
                prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="", lines=5, max_lines=5, scale=1)
                ref_emotion_text = gr.Textbox(
                    label=i18n("情绪/括注（来自训练样本，无数据留空）"),
                    value="",
                    interactive=False,
                    scale=1,
                )
            with gr.Column(scale=14):
                prompt_language = gr.Dropdown(
                    label=i18n("参考音频的语种"),
                    choices=list(dict_language.keys()),
                    value=i18n("中文"),
                )
                inp_refs = (
                    gr.File(
                        label=i18n(
                            "可选项：通过拖拽多个文件上传多个参考音频（建议同性），平均融合他们的音色。如不填写此项，音色由左侧单个参考音频控制。如是微调模型，建议参考音频全部在微调训练集音色内，底模不用管。"
                        ),
                        file_count="multiple",
                    )
                    if model_version not in v3v4set
                    else gr.File(
                        label=i18n(
                            "可选项：通过拖拽多个文件上传多个参考音频（建议同性），平均融合他们的音色。如不填写此项，音色由左侧单个参考音频控制。如是微调模型，建议参考音频全部在微调训练集音色内，底模不用管。"
                        ),
                        file_count="multiple",
                        visible=False,
                    )
                )
                sample_steps = (
                    gr.Radio(
                        label=i18n("采样步数,如果觉得电,提高试试,如果觉得慢,降低试试"),
                        value=32 if model_version == "v3" else 8,
                        choices=[4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
                        visible=True,
                    )
                    if model_version in v3v4set
                    else gr.Radio(
                        label=i18n("采样步数,如果觉得电,提高试试,如果觉得慢,降低试试"),
                        choices=[4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
                        visible=False,
                        value=32 if model_version == "v3" else 8,
                    )
                )
                if_sr_Checkbox = gr.Checkbox(
                    label=i18n("v3输出如果觉得闷可以试试开超分"),
                    value=False,
                    interactive=True,
                    show_label=True,
                    visible=False if model_version != "v3" else True,
                )
        gr.Markdown(html_center(i18n("*请填写需要合成的目标文本和语种模式"), "h3"))
        with gr.Row():
            with gr.Column(scale=13):
                text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=26, max_lines=26)
            with gr.Column(scale=7):
                text_language = gr.Dropdown(
                    label=i18n("需要合成的语种") + i18n(".限制范围越小判别效果越好。"),
                    choices=list(dict_language.keys()),
                    value=i18n("中文"),
                    scale=1,
                )
                how_to_cut = gr.Dropdown(
                    label=i18n("怎么切"),
                    choices=[
                        i18n("不切"),
                        i18n("凑四句一切"),
                        i18n("凑50字一切"),
                        i18n("按中文句号。切"),
                        i18n("按英文句号.切"),
                        i18n("按标点符号切"),
                    ],
                    value=i18n("凑四句一切"),
                    interactive=True,
                    scale=1,
                )
                gr.Markdown(value=html_center(i18n("语速调整，高为更快")))
                if_freeze = gr.Checkbox(
                    label=i18n("是否直接对上次合成结果调整语速和音色。防止随机性。"),
                    value=False,
                    interactive=True,
                    show_label=True,
                    scale=1,
                )
                with gr.Row():
                    speed = gr.Slider(
                        minimum=0.6, maximum=1.65, step=0.05, label=i18n("语速"), value=1, interactive=True, scale=1
                    )
                    pause_second_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        step=0.01,
                        label=i18n("句间停顿秒数"),
                        value=0.3,
                        interactive=True,
                        scale=1,
                    )
                gr.Markdown(html_center(i18n("GPT采样参数(无参考文本时不要太低。不懂就用默认)：")))
                top_k = gr.Slider(
                    minimum=1, maximum=100, step=1, label=i18n("top_k"), value=15, interactive=True, scale=1
                )
                top_p = gr.Slider(
                    minimum=0, maximum=1, step=0.05, label=i18n("top_p"), value=1, interactive=True, scale=1
                )
                temperature = gr.Slider(
                    minimum=0, maximum=1, step=0.05, label=i18n("temperature"), value=1, interactive=True, scale=1
                )
            # with gr.Column():
            #     gr.Markdown(value=i18n("手工调整音素。当音素框不为空时使用手工音素输入推理，无视目标文本框。"))
            #     phoneme=gr.Textbox(label=i18n("音素框"), value="")
            #     get_phoneme_button = gr.Button(i18n("目标文本转音素"), variant="primary")
        with gr.Row():
            inference_button = gr.Button(value=i18n("合成语音"), variant="primary", size="lg", scale=25)
            use_cuda_graph_checkbox = gr.Checkbox(
                label="CUDA Graph " + i18n("加速"),
                value=cuda_graph_supported,
                interactive=True if torch.cuda.is_available() else False,
                show_label=True,
                scale=5,
                visible=False,
            )
            output = gr.Audio(label=i18n("输出的语音"), scale=14)

        inference_button.click(
            get_tts_wav,
            [
                inp_ref,
                prompt_text,
                prompt_language,
                text,
                text_language,
                how_to_cut,
                top_k,
                top_p,
                temperature,
                ref_text_free,
                speed,
                if_freeze,
                inp_refs,
                sample_steps,
                if_sr_Checkbox,
                pause_second_slider,
                use_cuda_graph_checkbox,
            ],
            [output],
        )
        SoVITS_dropdown.change(
            change_sovits_weights,
            [SoVITS_dropdown, prompt_language, text_language],
            [
                prompt_language,
                text_language,
                prompt_text,
                prompt_language,
                text,
                text_language,
                sample_steps,
                inp_refs,
                ref_text_free,
                if_sr_Checkbox,
                inference_button,
            ],
        )
        GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

        # ===== 新增事件绑定：训练角色选择 / 样本浏览 =====
        refresh_models_btn.click(
            fn=refresh_model_dropdown,
            inputs=[],
            outputs=[model_name_dropdown],
        )
        model_name_dropdown.change(
            fn=on_model_name_change,
            inputs=[model_name_dropdown],
            outputs=[ref_sample_dropdown, GPT_dropdown, SoVITS_dropdown],
        )
        ref_sample_dropdown.change(
            fn=on_ref_sample_change,
            inputs=[ref_sample_dropdown, model_name_dropdown],
            outputs=[inp_ref, prompt_text, ref_emotion_text],
        )
        # 页面加载时自动扫描模型列表
        app.load(fn=refresh_model_dropdown, inputs=[], outputs=[model_name_dropdown])


        # gr.Markdown(value=i18n("文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。合成会根据文本的换行分开合成再拼起来。"))
        # with gr.Row():
        #     text_inp = gr.Textbox(label=i18n("需要合成的切分前文本"), value="")
        #     button1 = gr.Button(i18n("凑四句一切"), variant="primary")
        #     button2 = gr.Button(i18n("凑50字一切"), variant="primary")
        #     button3 = gr.Button(i18n("按中文句号。切"), variant="primary")
        #     button4 = gr.Button(i18n("按英文句号.切"), variant="primary")
        #     button5 = gr.Button(i18n("按标点符号切"), variant="primary")
        #     text_opt = gr.Textbox(label=i18n("切分后文本"), value="")
        #     button1.click(cut1, [text_inp], [text_opt])
        #     button2.click(cut2, [text_inp], [text_opt])
        #     button3.click(cut3, [text_inp], [text_opt])
        #     button4.click(cut4, [text_inp], [text_opt])
        #     button5.click(cut5, [text_inp], [text_opt])
        # gr.Markdown(html_center(i18n("后续将支持转音素、手工修改音素、语音合成分步执行。")))

if __name__ == "__main__":
    app.queue().launch(  # concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=infer_ttswebui,
        # quiet=True,
    )
