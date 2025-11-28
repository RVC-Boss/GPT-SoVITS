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
import soundfile  # 新增导入

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
    global vq_model, hps, version, model_version, dict_language, if_lora_v3
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


def change_gpt_weights(gpt_path):
    if "！" in gpt_path or "!" in gpt_path:
        gpt_path = name2gpt_path[gpt_path]
    global hz, max_sec, t2s_model, config
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
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                gr.Warning(i18n("参考音频在3~10秒范围外，请更换！"))
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
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


import uuid
import shutil
from pydub import AudioSegment

TEMP_FOLDER = "TEMP"  # 临时文件夹路径
os.makedirs(TEMP_FOLDER, exist_ok=True)

def clean_temp_folder_on_startup():
    """启动时清理临时文件夹"""
    try:
        if os.path.exists(TEMP_FOLDER):
            shutil.rmtree(TEMP_FOLDER)
            os.makedirs(TEMP_FOLDER, exist_ok=True)
            print("启动时已清理临时文件夹")
        else:
            os.makedirs(TEMP_FOLDER, exist_ok=True)
            print("临时文件夹已创建")
    except Exception as e:
        print(f"启动时清理临时文件夹失败: {str(e)}")

def split_text_by_punctuation(text, period_pause=0.3, comma_pause=0.15):
    """改进的文本分割函数"""
    if not text or not isinstance(text, str):
        print("收到空或非字符串文本输入")
        return []
    
    segments = []
    current_segment = ""
    punctuation_marks = ['，', ',', '。', '.']
    
    for char in text:
        current_segment += char
        if char in punctuation_marks:
            # 根据标点类型设置停顿时间
            pause = period_pause if char in ['。', '.'] else comma_pause
            segments.append({
                "text": current_segment.strip(),
                "pause": pause,
                "punctuation": char
            })
            current_segment = ""
    
    if current_segment:
        segments.append({
            "text": current_segment.strip(),
            "pause": comma_pause,  # 默认使用非句号停顿时间
            "punctuation": ""
        })
    
    print(f"分割结果: {[seg['text'] for seg in segments]}")
    return segments

def generate_segment_audio(segment_data, ref_wav_path, prompt_text, prompt_language, text_language, top_k, top_p, temperature):
    """增强的音频生成函数"""
    try:
        if not os.path.exists(ref_wav_path):
            raise FileNotFoundError(f"参考音频不存在: {ref_wav_path}")
        
        # 生成音频
        sr, audio_data = next(get_tts_wav(
            ref_wav_path=ref_wav_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            text=segment_data["text"],
            text_language=text_language,
            how_to_cut=i18n("不切"),
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pause_second=0  # 不在内部添加停顿
        ))
        
        # 这里不再生成ID，由调用者提供
        temp_path = os.path.join(TEMP_FOLDER, "temp_generate.wav")
        soundfile.write(temp_path, audio_data, sr)
        
        # 添加停顿
        audio = AudioSegment.from_wav(temp_path)
        pause = AudioSegment.silent(duration=int(segment_data["pause"]*1000))
        final_audio = audio + pause
        final_audio.export(temp_path, format="wav")
        
        return {
            "success": True,
            "audio_path": temp_path,
            "text": segment_data["text"],
            "pause": segment_data["pause"],
            "message": "生成成功"
        }
    except Exception as e:
        print(f"生成片段失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "audio_path": None,
            "text": segment_data["text"],
            "pause": segment_data["pause"],
            "message": f"生成失败: {str(e)}"
        }

def process_all_segments(text, ref_wav_path, prompt_text, prompt_language, text_language, 
                        top_k, top_p, temperature, period_pause, comma_pause):
    """完整处理流程"""
    # 输入验证
    if not text or not isinstance(text, str):
        error_msg = "输入文本无效"
        print(error_msg)
        return [[1, error_msg, "错误"]], None
    
    if not os.path.exists(ref_wav_path):
        error_msg = f"参考音频不存在: {ref_wav_path}"
        print(error_msg)
        return [[1, error_msg, "错误"]], None
    
    # 处理分段
    segments = split_text_by_punctuation(text, period_pause, comma_pause)
    if not segments:
        error_msg = "无法分割文本"
        print(error_msg)
        return [[1, error_msg, "错误"]], None
    
    results = []
    audio_files = []
    
    # 修改这里：使用enumerate从1开始编号，而不是基于文件夹内容
    for i, segment in enumerate(segments, 1):
        result = generate_segment_audio(
            segment, ref_wav_path, prompt_text,
            prompt_language, text_language, top_k, top_p, temperature
        )
        
        # 更新结果中的segment_id
        result["segment_id"] = f"{i}temp"
        if result["success"] and result["audio_path"]:
            # 重命名文件以匹配新的编号
            new_path = os.path.join(TEMP_FOLDER, f"{result['segment_id']}.wav")
            os.rename(result["audio_path"], new_path)
            result["audio_path"] = new_path
            audio_files.append(new_path)
        
        results.append(result)
        print(f"处理进度: {i}/{len(segments)} - {result['message']}")

    # 准备显示数据
    df_data = []
    for i, result in enumerate(results, 1):
        df_data.append([
            f"{i}temp",
            result["text"],
            result["message"]
        ])
    
    first_audio = audio_files[0] if audio_files else None
    return df_data, first_audio

def regenerate_segment(segment_id, new_text, ref_wav_path, prompt_text, 
                      prompt_language, text_language, top_k, top_p, temperature,
                      period_pause, comma_pause):
    try:
        if not segment_id or not new_text:
            raise ValueError("缺少片段ID或新文本内容")
        
        # 从文件名解析原始停顿时间
        try:
            pause = 0.25 if segment_id.endswith(("。", ".")) else 0.1
        except:
            pause = 0.1  # 默认值
        
        is_period = segment_id.endswith(("。", "."))
        pause = period_pause if is_period else comma_pause
        
        segment_data = {
            "text": new_text,
            "pause": pause,
            "punctuation": "。" if is_period else "，"
        }
        result = generate_segment_audio(
            segment_data, ref_wav_path, prompt_text,
            prompt_language, text_language, top_k, top_p, temperature
        )
        
        # 更新文件
        if result["success"]:
            old_path = os.path.join(TEMP_FOLDER, f"{segment_id}.wav")
            if os.path.exists(old_path):
                os.remove(old_path)
            os.rename(result["audio_path"], old_path)
            result["audio_path"] = old_path
        
        return (
            result["audio_path"],
            segment_id,
            result["message"]
        )
    except Exception as e:
        print(f"重新生成片段失败: {str(e)}", exc_info=True)
        return None, segment_id, f"重新生成失败: {str(e)}"

def merge_all_segments():
    try:
        # 获取并按编号排序片段
        segments = sorted(
            [f for f in os.listdir(TEMP_FOLDER) if f.endswith(".wav") and f != "final_output.wav"],
            key=lambda x: int(x.split("temp")[0])
        )
        if not segments:
            raise ValueError("没有找到可合并的音频片段")
        
        combined = AudioSegment.empty()
        for seg in segments:
            seg_path = os.path.join(TEMP_FOLDER, seg)
            audio = AudioSegment.from_wav(seg_path)
            combined += audio
        
        # 保存最终结果
        output_path = os.path.join(TEMP_FOLDER, "final_output.wav")
        combined.export(output_path, format="wav")
        
        print(f"成功合并 {len(segments)} 个片段")
        return output_path, "合并成功"
    except Exception as e:
        print(f"合并片段失败: {str(e)}", exc_info=True)
        return None, f"合并失败: {str(e)}"

def clean_temp_files():
    """清理临时文件函数"""
    try:
        for filename in os.listdir(TEMP_FOLDER):
            file_path = os.path.join(TEMP_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"删除文件 {file_path} 失败: {e}")
        return "临时文件已清理"
    except Exception as e:
        return f"清理失败: {str(e)}"

def on_segment_select(df, evt: gr.SelectData):
    """当选择分段列表中的项目时更新显示"""
    if evt.index:
        selected_row = df.iloc[evt.index[0]]
        audio_path = os.path.join(TEMP_FOLDER, f"{selected_row['编号']}.wav")
        return (
            selected_row["编号"],
            selected_row["文本内容"],
            audio_path if os.path.exists(audio_path) else None
        )
    return "1temp", "", None

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
# ======================== 合并功能实现 ========================
def merge_selected_segments(merge_range, segment_list_data, ref_wav_path, prompt_text, 
                           prompt_language, text_language, top_k, top_p, temperature,
                           pause_period, pause_comma):
    """合并选中的句子并立即生成新音频"""
    try:
        if not merge_range:
            return segment_list_data, "请输入合并范围（例如：1-3）", None
        
        # 解析合并范围
        start, end = map(int, merge_range.split('-'))
        if start <= 0 or end <= 0 or start > end:
            return segment_list_data, "无效的合并范围", None
        
        # 检查范围是否有效
        if end > len(segment_list_data):
            return segment_list_data, f"结束编号 {end} 超过总段数 {len(segment_list_data)}", None
        
        # === 第一步：收集要删除的文件并立即删除 ===
        files_to_delete = []
        for i in range(start-1, end):
            file_id = segment_list_data.iloc[i, 0]
            file_path = os.path.join(TEMP_FOLDER, f"{file_id}.wav")
            if os.path.exists(file_path):
                os.remove(file_path)
                files_to_delete.append(file_id)
        
        # === 第二步：合并文本 ===
        merged_text = ""
        for i in range(start-1, end):
            merged_text += segment_list_data.iloc[i, 1]  # 文本内容在第二列
        
        # 确定合并后的停顿类型（取最后一个片段的标点）
        last_punctuation = segment_list_data.iloc[end-1, 1][-1] if segment_list_data.iloc[end-1, 1] else ""
        is_period = last_punctuation in ["。", "."]
        pause = pause_period if is_period else pause_comma
        
        # === 第三步：立即生成合并后的音频 ===
        segment_data = {
            "text": merged_text,
            "pause": pause,
            "punctuation": last_punctuation
        }
        # 生成音频
        result = generate_segment_audio(
            segment_data, ref_wav_path, prompt_text,
            prompt_language, text_language, top_k, top_p, temperature
        )
        if not result["success"]:
            return segment_list_data, result["message"], None
        
        # === 第四步：构建新的分段列表 ===
        # 创建新的合并条目 - 使用起始编号
        new_id = f"{start}temp"
        merged_entry = [new_id, merged_text, "已生成"]
        
        # 移动生成的音频到正确位置
        new_path = os.path.join(TEMP_FOLDER, f"{new_id}.wav")
        shutil.move(result["audio_path"], new_path)
        
        # 构建新的分段列表
        new_segment_list = []
        
        # 添加合并前的部分
        if start > 1:
            new_segment_list.extend(segment_list_data.iloc[:start-1].values.tolist())
        
        # 添加合并条目
        new_segment_list.append(merged_entry)
        
        # 添加合并后的部分
        if end < len(segment_list_data):
            new_segment_list.extend(segment_list_data.iloc[end:].values.tolist())
        
        # === 第五步：重新编号 ===
        reindexed_list = []
        new_id_counter = 1
        for segment in new_segment_list:
            old_id = segment[0]
            # 为新列表生成连续编号
            new_id = f"{new_id_counter}temp"
            
            # 重命名文件（如果存在）
            old_path = os.path.join(TEMP_FOLDER, f"{old_id}.wav")
            new_path = os.path.join(TEMP_FOLDER, f"{new_id}.wav")
            
            if os.path.exists(old_path) and old_id != new_id:
                os.rename(old_path, new_path)
            
            # 更新ID
            segment[0] = new_id
            reindexed_list.append(segment)
            new_id_counter += 1
        
        return reindexed_list, "合并成功", new_id
    
    except Exception as e:
        traceback.print_exc()
        return segment_list_data, f"合并失败: {str(e)}", None

# ======================== 实现拆分逻辑函数 ========================
def split_selected_segment(split_id, segment_list_data, ref_wav_path, prompt_text, 
                         prompt_language, text_language, top_k, top_p, temperature,
                         pause_period, pause_comma):
    """拆分选中的句子并重新生成 - 使用倒序重命名避免冲突"""
    try:
        if not split_id:
            return segment_list_data, "请输入要拆分的句子编号", None
        
        # 从文件名解析原始停顿时间
        try:
            pause = 0.25 if split_id.endswith(("。", ".")) else 0.1
        except:
            pause = 0.1  # 默认值
        
        # 查找要拆分的句子
        df = segment_list_data
        target_row = None
        target_idx = None
        
        # 查找匹配的句子
        for idx, row in enumerate(df.itertuples()):
            if str(row[1]) == split_id:  # 第一列是ID
                target_row = row
                target_idx = idx
                break
        
        if target_row is None:
            return segment_list_data, f"未找到编号为 {split_id} 的句子", None
        
        # 获取原始文本
        original_text = target_row[2]  # 第二列是文本
        
        # 根据标点符号拆分文本
        segments = []
        current_segment = ""
        punctuation_marks = ['，', ',', '。', '.', '？', '?', '！', '!', '；', ';']
        
        for char in original_text:
            current_segment += char
            if char in punctuation_marks:
                # 根据标点类型设置停顿时间
                pause = pause_period if char in ['。', '.', '？', '?', '！', '!'] else pause_comma
                segments.append({
                    "text": current_segment.strip(),
                    "pause": pause,
                    "punctuation": char
                })
                current_segment = ""
        
        if current_segment:
            segments.append({
                "text": current_segment.strip(),
                "pause": pause_comma,  # 默认使用非句号停顿时间
                "punctuation": ""
            })
        
        if len(segments) <= 1:
            return segment_list_data, "句子无法拆分（没有标点符号）", None
        
        # 计算拆分后新增的句子数量
        num_new_segments = len(segments)
        offset = num_new_segments - 1  # 拆分后增加的句子数
        
        # === 第一步：删除原始音频文件 ===
        original_path = os.path.join(TEMP_FOLDER, f"{split_id}.wav")
        if os.path.exists(original_path):
            os.remove(original_path)
        
        # === 第二步：倒序重命名后续句子避免冲突 ===
        new_segment_list = []
        
        # 添加拆分前的句子
        for i in range(target_idx):
            new_segment_list.append(df.iloc[i].tolist())
        
        # 倒序重命名从最后一个句子开始，避免冲突
        total_segments = len(df)
        
        # 从最后一个句子开始倒序遍历
        for i in range(total_segments - 1, target_idx, -1):
            old_id = df.iloc[i, 0]
            old_num = int(old_id.replace("temp", ""))
            new_id_num = old_num + offset
            new_id = f"{new_id_num}temp"
            
            # 重命名音频文件
            old_path = os.path.join(TEMP_FOLDER, f"{old_id}.wav")
            new_path = os.path.join(TEMP_FOLDER, f"{new_id}.wav")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
            
            # 更新列表条目
            new_segment_list.append([new_id, df.iloc[i, 1], df.iloc[i, 2]])
        
        # === 第三步：添加拆分后的新句子 ===
        for i, segment in enumerate(segments):
            new_id = f"{target_idx+1+i}temp"  # 新ID从原始位置开始
            new_segment_list.append([new_id, segment["text"], "待生成"])
        
        # === 第四步：重新排序列表 ===
        # 按照ID中的数字排序
        new_segment_list.sort(key=lambda x: int(x[0].replace("temp", "")))
        
        # === 第五步：生成拆分后的新句子 ===
        for i, segment in enumerate(segments):
            segment_id = f"{target_idx+1+i}temp"
            result = generate_segment_audio(
                segment, ref_wav_path, prompt_text,
                prompt_language, text_language, top_k, top_p, temperature
            )
            
            if result["success"]:
                # 更新文件
                new_path = os.path.join(TEMP_FOLDER, f"{segment_id}.wav")
                if os.path.exists(result["audio_path"]):
                    os.rename(result["audio_path"], new_path)
                
                # 更新状态
                for j, item in enumerate(new_segment_list):
                    if item[0] == segment_id:
                        new_segment_list[j][2] = "已生成"
                        break
        
        return new_segment_list, f"成功拆分为 {num_new_segments} 个句子", f"{target_idx+1}temp"
    
    except Exception as e:
        traceback.print_exc()
        return segment_list_data, f"拆分失败: {str(e)}", None



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
        gr.Markdown(html_center(i18n("*请上传并填写参考信息"), "h3"))
        with gr.Row():
            inp_ref = gr.Audio(label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath", scale=13)
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



        # ======================== 插入分段合成UI开始 ========================
    with gr.Tab(i18n("分段合成模式")):
        with gr.Row():
            with gr.Column():
                segmented_text = gr.Textbox(label=i18n("需要分段的文本"), lines=10, value="")
                segment_button = gr.Button(i18n("分割文本并生成所有片段"), variant="primary")
                segmented_output = gr.Audio(label=i18n("当前选中片段"), interactive=False)
                
                with gr.Row():
                    segment_index = gr.Textbox(label=i18n("片段编号"), interactive=False, visible=False)
                    new_segment_text = gr.Textbox(label=i18n("新文本内容"), lines=2, max_lines=4)
                    regenerate_button = gr.Button(i18n("重新生成当前片段"), variant="primary")
                with gr.Row():
                    pause_period = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        step=0.05, 
                        label=i18n("句号停顿时间(秒)"), 
                        value=0.3,
                        interactive=True
                    )
                    pause_comma = gr.Slider(
                        minimum=0.1, 
                        maximum=0.5, 
                        step=0.05, 
                        label=i18n("非句号停顿时间(秒)"), 
                        value=0.15,
                        interactive=True
                    )
                with gr.Row():
                    clean_button = gr.Button(i18n("清理临时文件"), variant="secondary", scale=1)
                    confirm_button = gr.Button(i18n("确认并合并所有片段"), variant="primary", scale=2)
                
                final_output = gr.Audio(label=i18n("最终合成结果"), interactive=False)
                segment_status = gr.Textbox(label=i18n("状态"), interactive=False)
                # 添加合并功能
                with gr.Row():
                    merge_range = gr.Textbox(label=i18n("合并范围（例如：1-3）"), scale=3)
                    merge_button = gr.Button(i18n("合并句子"), variant="primary", scale=1)
                # 添加拆分功能控件
                with gr.Row():
                    split_index = gr.Textbox(label=i18n("要拆分的句子编号"), scale=1)
                    split_button = gr.Button(i18n("拆分句子"), variant="primary", scale=1)
            with gr.Column():
                segment_list = gr.Dataframe(
                    headers=["编号", "文本内容", "状态"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    label=i18n("分段列表"),
                    value=[]
                )
                # 在分段合成UI部分添加以下代码（大约在line 3200附近）

    # ======================== 插入分段合成UI结束 ========================

    # ======================== 插入分段合成事件绑定开始 ========================
    # 分割文本并生成所有片段
    segment_button.click(
        process_all_segments,
        inputs=[
            segmented_text, 
            inp_ref, 
            prompt_text, 
            prompt_language, 
            text_language, 
            top_k, 
            top_p, 
            temperature,
            pause_period,  # 新增句号停顿参数
            pause_comma    # 新增非句号停顿参数
        ],
        outputs=[segment_list, segmented_output]
    )
    # 重新生成当前片段
    regenerate_button.click(
        regenerate_segment,
        inputs=[
            segment_index,  # 第一个参数应该是segment_id
            new_segment_text, 
            inp_ref, 
            prompt_text, 
            prompt_language, 
            text_language, 
            top_k, 
            top_p, 
            temperature,
            pause_period,
            pause_comma
        ],
        outputs=[segmented_output, segment_index, segment_status]
    )

    # 合并所有片段
    confirm_button.click(
        merge_all_segments,
        inputs=[],
        outputs=[final_output, segment_status]
    )

    # 清理临时文件
    clean_button.click(
        clean_temp_files,
        inputs=[],
        outputs=[segment_status]
    )

    # 当选择分段列表中的项目时
    segment_list.select(
        fn=on_segment_select,
        inputs=[segment_list],
        outputs=[segment_index, new_segment_text, segmented_output],
    )
    merge_button.click(
        fn=merge_selected_segments,
        inputs=[
            merge_range, 
            segment_list,
            inp_ref,
            prompt_text,
            prompt_language,
            text_language,
            top_k,
            top_p,
            temperature,
            pause_period,
            pause_comma
        ],
        outputs=[segment_list, segment_status, segment_index]
    )
    # ======================== 添加拆分按钮的事件绑定 ========================
    split_button.click(
        fn=split_selected_segment,
        inputs=[
            split_index, 
            segment_list,
            inp_ref,
            prompt_text,
            prompt_language,
            text_language,
            top_k,
            top_p,
            temperature,
            pause_period,
            pause_comma
        ],
        outputs=[segment_list, segment_status, segment_index]
    )
    # ======================== 插入分段合成事件绑定结束 ========================




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

