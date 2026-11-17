#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
- 通过 GET 和 POST 请求提供 TTS 推理接口 (`/`)，支持默认参考音频和参数调整。
- 新增 `/ttsrole` 接口，支持基于角色的 TTS 推理，动态加载角色模型和参考音频，同时支持 GET 和 POST 请求。
- 支持更换默认参考音频 (`/change_refer`) 和模型权重 (`/set_model`)。
- 提供控制接口 (`/control`) 用于重启或退出服务。
- 支持多语言文本处理（中文、英文、日文、韩文等）及自动语言切分。
- 支持多种音频格式（wav, ogg, aac）和数据类型（int16, int32）。
- 支持通过 POST 请求动态切换模型版本（v2 或 v3）。

使用方法：
1. 安装依赖：
   pip install -r requirements.txt
2. 配置环境：
   - 确保 GPT 和 SoVITS 模型文件已准备好。
   - 可选：设置默认参考音频路径、文本和语言。
3. 运行服务：
   python api_role_v3.py -s "path/to/sovits.pth" -g "path/to/gpt.ckpt" -dr "ref.wav" -dt "参考文本" -dl "zh" -p 9880

参数说明：
命令行参数：
- -s, --sovits_path: SoVITS 模型路径（默认从 config 获取）。
- -g, --gpt_path: GPT 模型路径（默认从 config 获取）。
- -dr, --default_refer_path: 默认参考音频路径。
- -dt, --default_refer_text: 默认参考音频文本。
- -dl, --default_refer_language: 默认参考音频语言（zh, en, ja, ko 等）。
- -d, --device: 设备（cuda 或 cpu，默认从 config 获取）。
- -a, --bind_addr: 绑定地址（默认 0.0.0.0）。
- -p, --port: 端口（默认 9880）。
- -fp, --full_precision: 使用全精度（覆盖默认）。
- -hp, --half_precision: 使用半精度（覆盖默认）。
- -sm, --stream_mode: 流式模式（close 或 normal，默认 close）。
- -mt, --media_type: 音频格式（wav, ogg, aac，默认 wav）。
- -st, --sub_type: 数据类型（int16 或 int32，默认 int16）。
- -cp, --cut_punc: 文本切分符号（默认空）。
- -hb, --hubert_path: HuBERT 模型路径（默认从 config 获取）。
- -b, --bert_path: BERT 模型路径（默认从 config 获取）。

接口参数（/）：
- refer_wav_path: 参考音频路径（可选）。
- prompt_text: 参考音频文本（可选）。
- prompt_language: 参考音频语言（可选）。
- text: 待合成文本（必填）。
- text_language: 目标文本语言（可选，默认 auto）。
- cut_punc: 文本切分符号（可选）。
- top_k: Top-K 采样值（默认 15）。
- top_p: Top-P 采样值（默认 1.0）。
- temperature: 温度值（默认 1.0）。
- speed: 语速因子（默认 1.0）。
- inp_refs: 辅助参考音频路径列表（默认空）。
- sample_steps: 采样步数（默认 32，限定 [4, 8, 16, 32]）。
- if_sr: 是否超分（默认 False）。

接口参数（/ttsrole）：
- text: 待合成文本（必填）。
- role: 角色名称（必填）。
- text_language: 目标文本语言（默认 auto）。
- ref_audio_path: 参考音频路径（可选）。
- prompt_text: 参考音频文本（可选）。
- prompt_language: 参考音频语言（可选）。
- emotion: 情感标签（可选）。
- top_k: Top-K 采样值（默认 15）。
- top_p: Top-P 采样值（默认 0.6）。
- temperature: 温度值（默认 0.6）。
- speed: 语速因子（默认 1.0）。
- inp_refs: 辅助参考音频路径列表（默认空）。
- sample_steps: 采样步数（默认 32，限定 [4, 8, 16, 32]）。
- if_sr: 是否超分（默认 False）。
- version: 模型版本（可选，v2 或 v3，POST 请求支持动态切换）。

### 完整请求示例 (/ttsrole POST)
{
    "text": "你好",                     # str, 必填, 要合成的文本内容
    "role": "role1",                   # str, 必填, 角色名称，决定使用 roles/{role} 中的配置和音频
    "emotion": "开心",                  # str, 可选, 情感标签，用于从 roles/{role}/reference_audios 中选择音频
    "text_lang": "auto",               # str, 可选, 默认 "auto", 文本语言，"auto" 时根据 emotion 或角色目录动态选择
    "ref_audio_path": "/path/to/ref.wav",  # str, 可选, 参考音频路径，若提供则优先使用，跳过自动选择
    "aux_ref_audio_paths": ["/path1.wav", "/path2.wav"],  # List[str], 可选, 辅助参考音频路径，用于多说话人融合
    "prompt_lang": "ja",               # str, 可选, 提示文本语言，若提供 ref_audio_path 则需指定，"auto" 模式下动态选择
    "prompt_text": "こんにちは",       # str, 可选, 提示文本，与 ref_audio_path 配对使用，自动选择时从文件或文件名生成
    "top_k": 10,                       # int, 可选, Top-K 采样值，覆盖 inference.top_k
    "top_p": 0.8,                      # float, 可选, Top-P 采样值，覆盖 inference.top_p
    "temperature": 1.0,                # float, 可选, 温度值，覆盖 inference.temperature
    "text_split_method": "cut5",       # str, 可选, 文本分割方法，覆盖 inference.text_split_method, 具体见text_segmentation_method.py
    "batch_size": 2,                   # int, 可选, 批处理大小，覆盖 inference.batch_size
    "batch_threshold": 0.75,           # float, 可选, 批处理阈值，覆盖 inference.batch_threshold
    "split_bucket": true,              # bool, 可选, 是否按桶分割，覆盖 inference.split_bucket
    "speed_factor": 1.2,               # float, 可选, 语速因子，覆盖 inference.speed_factor
    "fragment_interval": 0.3,          # float, 可选, 片段间隔（秒），覆盖 inference.fragment_interval
    "seed": 42,                        # int, 可选, 随机种子，覆盖 seed
    "media_type": "wav",               # str, 可选, 默认 "wav", 输出格式，支持 "wav", "raw", "ogg", "aac"
    "streaming_mode": false,           # bool, 可选, 默认 false, 是否流式返回
    "parallel_infer": true,            # bool, 可选, 默认 true, 是否并行推理
    "repetition_penalty": 1.35,        # float, 可选, 重复惩罚值，覆盖 inference.repetition_penalty
    "version": "v2",                   # str, 可选, 配置文件版本，覆盖 version，动态切换 v2 或 v3
    "languages": ["zh", "ja", "en"],   # List[str], 可选, 支持的语言列表，覆盖 languages
    "bert_base_path": "/path/to/bert", # str, 可选, BERT 模型路径，覆盖 bert_base_path
    "cnhuhbert_base_path": "/path/to/hubert",  # str, 可选, HuBERT 模型路径，覆盖 cnhuhbert_base_path
    "device": "cpu",                   # str, 可选, 统一设备，覆盖 device
    "is_half": true,                   # bool, 可选, 是否使用半精度，覆盖 is_half
    "t2s_weights_path": "/path/to/gpt.ckpt",  # str, 可选, GPT 模型路径，覆盖 t2s_weights_path
    "vits_weights_path": "/path/to/sovits.pth",  # str, 可选, SoVITS 模型路径，覆盖 vits_weights_path
    "t2s_model_path": "/path/to/gpt.ckpt",  # str, 可选, GPT 模型路径（与 t2s_weights_path 同义）
    "t2s_model_device": "cpu",         # str, 可选, GPT 模型设备，覆盖 t2s_model.device，默认检测显卡
    "vits_model_path": "/path/to/sovits.pth",  # str, 可选, SoVITS 模型路径（与 vits_weights_path 同义）
    "vits_model_device": "cpu"         # str, 可选, SoVITS 模型设备，覆盖 vits_model.device，默认检测显卡
}

### 参数必要性和优先级
- 必填参数:
  - /ttsrole: text, role
  - /tts: text, ref_audio_path, prompt_lang
- 可选参数: 其他均为可选，默认值从 roles/{role}/tts_infer.yaml 或 GPT_SoVITS/configs/tts_infer.yaml 获取
- 优先级: POST 请求参数 > roles/{role}/tts_infer.yaml > 默认 GPT_SoVITS/configs/tts_infer.yaml

### 目录结构
GPT-SoVITS-roleapi/
├── api_role_v3.py                         # 本文件, API 主程序
├── GPT_SoVITS/                    # GPT-SoVITS 核心库
│   └── configs/
│       └── tts_infer.yaml         # 默认配置文件
├── roles/                         # 角色配置目录
│   ├── role1/                     # 示例角色 role1
│   │   ├── tts_infer.yaml         # 角色配置文件（可选）
│   │   ├── model.ckpt             # GPT 模型（可选）
│   │   ├── model.pth              # SoVITS 模型（可选）
│   │   └── reference_audios/      # 角色参考音频目录
│   │       ├── zh/
│   │       │   ├── 【开心】voice1.wav
│   │       │   ├── 【开心】voice1.txt
│   │       ├── ja/
│   │       │   ├── 【开心】voice2.wav
│   │       │   ├── 【开心】voice2.txt
│   ├── role2/
│   │   ├── tts_infer.yaml
│   │   ├── model.ckpt
│   │   ├── model.pth
│   │   └── reference_audios/
│   │       ├── zh/
│   │       │   ├── 【开心】voice1.wav
│   │       │   ├── 【开心】voice1.txt
│   │       │   ├── 【悲伤】asdafasdas.wav
│   │       │   ├── 【悲伤】asdafasdas.txt
│   │       ├── ja/
│   │       │   ├── 【开心】voice2.wav
│   │       │   ├── 【开心】voice2.txt

### text_lang, prompt_lang, prompt_text 选择逻辑 (/ttsrole)
1. text_lang 选择逻辑:
   - 默认值: "auto"
   - 如果请求未提供 text_lang，视为 "auto"
   - 当 text_lang = "auto" 且存在 emotion 参数：
     - 从 roles/{role}/reference_audios 下所有语言文件夹中查找以 "【emotion】" 开头的音频
     - 随机选择一个匹配的音频，语言由音频所在文件夹确定
   - 当 text_lang 指定具体语言（如 "zh"）：
     - 从 roles/{role}/reference_audios/{text_lang} 中选择音频
     - 如果指定语言无匹配音频，则尝试其他语言文件夹
2. prompt_lang 选择逻辑:
   - 如果提供了 ref_audio_path，则需显式指定 prompt_lang
   - 如果未提供 ref_audio_path 且 text_lang = "auto" 且存在 emotion：
     - prompt_lang = 随机选择的音频所在语言文件夹名（如 "zh" 或 "ja"）
   - 如果未提供 ref_audio_path 且 text_lang 指定具体语言：
     - prompt_lang = text_lang（如 "zh"）
     - 如果 text_lang 无匹配音频，则为随机选择的音频所在语言
3. prompt_text 选择逻辑:
   - 如果提供了 ref_audio_path（如 "/path/to/ref.wav"）：
     - 检查文件名是否包含 "【xxx】" 前缀：
       - 如果有（如 "【开心】abc.wav"）：
         - 若存在对应 .txt 文件（如 "【开心】abc.txt"），prompt_text = .txt 文件内容
         - 若无对应 .txt 文件，prompt_text = "abc"（去掉 "【开心】" 和 ".wav" 的部分）
       - 如果无 "【xxx】" 前缀：
         - 若存在对应 .txt 文件（如 "ref.txt"），prompt_text = .txt 文件内容
         - 若无对应 .txt 文件，prompt_text = "ref"（去掉 ".wav" 的部分）
   - 如果未提供 ref_audio_path：
     - 从 roles/{role}/reference_audios 中选择音频（基于 text_lang 和 emotion）：
       - 优先匹配 "【emotion】" 前缀的音频（如 "【开心】voice1.wav"）
       - 若存在对应 .txt 文件（如 "【开心】voice1.txt"），prompt_text = .txt 文件内容
       - 若无对应 .txt 文件，prompt_text = "voice1"（去掉 "【开心】" 和 ".wav" 的部分）
       - 未匹配 emotion 则随机选择一个音频，逻辑同上

### 讲解
1. 必填参数:
   - /ttsrole: text, role
   - /tts: text, ref_audio_path, prompt_lang
2. 音频选择 (/ttsrole):
   - 若提供 ref_audio_path，则使用它
   - 否则根据 role、text_lang、emotion 从 roles/{role}/reference_audios 中选择
   - text_lang = "auto" 时，若有 emotion，则跨语言匹配 "【emotion】" 前缀音频
   - emotion 匹配 "【emotion】" 前缀音频，未匹配则随机选择
3. 设备选择:
   - 默认尝试检测显卡（torch.cuda.is_available()），若可用则用 "cuda"，否则 "cpu"
   - 若缺少 torch 依赖或检测失败，回退到 "cpu"
   - POST 参数 device, t2s_model_device, vits_model_device 可强制指定设备，优先级最高
4. 配置文件:
   - 默认加载 GPT_SoVITS/configs/tts_infer.yaml
   - 若 roles/{role}/tts_infer.yaml 存在且未被请求参数覆盖，则使用它 (/ttsrole)
   - 请求参数（如 top_k, bert_base_path）覆盖所有配置文件
5. 返回格式:
   - 成功时返回音频流 (Response 或 StreamingResponse)
   - 失败时返回 JSON，包含错误消息和可能的异常详情
6. 运行:
   - python api_role_v3.py -a 127.0.0.1 -p 9880
   - 检查启动日志确认设备
7. 模型版本切换:
   - POST 请求中通过 "version" 参数指定 "v2" 或 "v3"，动态影响推理逻辑。
"""

import argparse
import os
import re
import sys
import signal
from time import time as ttime
import torch
import torchaudio
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from feature_extractor import cnhubert
from io import BytesIO
from module.models import SynthesizerTrn, SynthesizerTrnV3
from peft import LoraConfig, PeftModel, get_peft_model
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio
import config as global_config
import logging
import subprocess
import glob
from typing import Optional, List
from text.LangSegmenter import LangSegmenter
import random

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

# 日志配置
logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logger = logging.getLogger('uvicorn')

# 获取全局配置
g_config = global_config.Config()

# 默认参考音频类
class DefaultRefer:
    def __init__(self, path, text, language):
        self.path = path
        self.text = text
        self.language = language

    def is_ready(self) -> bool:
        return is_full(self.path, self.text, self.language)

def is_empty(*items):
    for item in items:
        if item is not None and item != "":
            return False
    return True

def is_full(*items):
    for item in items:
        if item is None or item == "":
            return False
    return True

# 角色和模型定义
class Speaker:
    def __init__(self, name, gpt, sovits, phones=None, bert=None, prompt=None):
        self.name = name
        self.gpt = gpt
        self.sovits = sovits
        self.phones = phones
        self.bert = bert
        self.prompt = prompt

class Sovits:
    def __init__(self, vq_model, hps):
        self.vq_model = vq_model
        self.hps = hps

class Gpt:
    def __init__(self, max_sec, t2s_model):
        self.max_sec = max_sec
        self.t2s_model = t2s_model

# 全局变量
speaker_list = {}
hz = 50
bigvgan_model = None

# BigVGAN 初始化
def init_bigvgan():
    global bigvgan_model
    from BigVGAN import bigvgan
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False
    )
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if is_half:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)

# 模型加载函数
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
def get_sovits_weights(sovits_path):
    path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
    is_exist_s2gv3 = os.path.exists(path_sovits_v3)

    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    if if_lora_v3 and not is_exist_s2gv3:
        logger.info("SoVITS V3 底模缺失，无法加载相应 LoRA 权重")

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if 'enc_p.text_embedding.weight' not in dict_s2['weight']:
        hps.model.version = "v2"
    elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"

    if model_version == "v3":
        hps.model.version = "v3"

    model_params_dict = vars(hps.model)
    if model_version != "v3":
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model_params_dict
        )
    else:
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model_params_dict
        )
        init_bigvgan()
    logger.info(f"模型版本: {hps.model.version}")
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    if not if_lora_v3:
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
    else:
        vq_model.load_state_dict(load_sovits_new(path_sovits_v3)["weight"], strict=False)
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        vq_model.eval()

    return Sovits(vq_model, hps)

def get_gpt_weights(gpt_path):
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    return Gpt(max_sec, t2s_model)

def change_gpt_sovits_weights(gpt_path, sovits_path):
    try:
        gpt = get_gpt_weights(gpt_path)
        sovits = get_sovits_weights(sovits_path)
    except Exception as e:
        return JSONResponse({"code": 400, "message": str(e)}, status_code=400)
    speaker_list["default"] = Speaker(name="default", gpt=gpt, sovits=sovits)
    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)

# 角色配置加载
def load_role_config(role, vits_weights_path=None, t2s_weights_path=None):
    role_dir = os.path.join(now_dir, "roles", role)
    if not os.path.exists(role_dir):
        return False
    gpt_path = t2s_weights_path or (glob.glob(os.path.join(role_dir, "*.ckpt"))[0] if glob.glob(os.path.join(role_dir, "*.ckpt")) else args.gpt_path)
    sovits_path = vits_weights_path or (glob.glob(os.path.join(role_dir, "*.pth"))[0] if glob.glob(os.path.join(role_dir, "*.pth")) else args.sovits_path)
    speaker_list[role] = Speaker(name=role, gpt=get_gpt_weights(gpt_path), sovits=get_sovits_weights(sovits_path))
    return True

# 参考音频选择
def select_ref_audio(role, text_language, emotion=None):
    audio_base_dir = os.path.join(now_dir, "roles", role, "reference_audios")
    if not os.path.exists(audio_base_dir):
        return None, None, None
    if text_language.lower() == "auto" and emotion:
        all_langs = [d for d in os.listdir(audio_base_dir) if os.path.isdir(os.path.join(audio_base_dir, d))]
        emotion_files = []
        for lang in all_langs:
            lang_dir = os.path.join(audio_base_dir, lang)
            emotion_files.extend(glob.glob(os.path.join(lang_dir, f"【{emotion}】*.*")))
        if emotion_files:
            audio_path = random.choice(emotion_files)
            txt_path = audio_path.rsplit(".", 1)[0] + ".txt"
            prompt_text = open(txt_path, "r", encoding="utf-8").read().strip() if os.path.exists(txt_path) else os.path.basename(audio_path).split("】")[1].rsplit(".", 1)[0]
            prompt_language = os.path.basename(os.path.dirname(audio_path))
            return audio_path, prompt_text, prompt_language
    lang_dir = os.path.join(audio_base_dir, text_language.lower())
    if os.path.exists(lang_dir):
        audio_files = glob.glob(os.path.join(lang_dir, f"【{emotion}】*.*" if emotion else "*.*"))
        if audio_files:
            audio_path = random.choice(audio_files)
            txt_path = audio_path.rsplit(".", 1)[0] + ".txt"
            prompt_text = open(txt_path, "r", encoding="utf-8").read().strip() if os.path.exists(txt_path) else os.path.basename(audio_path).rsplit(".", 1)[0]
            return audio_path, prompt_text, text_language.lower()
    return None, None, None

# BERT 和文本处理函数
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

def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)
    return bert

from text import chinese
def get_phones_and_bert(text, language, version, final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "all_zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext, "zh", version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "all_yue" and re.search(r'[A-Za-z]', formattext):
            formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(formattext, "yue", version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist = []
        langlist = []
        if language == "auto":
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
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])
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
        norm_text = ''.join(norm_text_list)
    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)
    return phones, bert.to(torch.float16 if is_half else torch.float32), norm_text

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

def get_spepc(hps, filename):
    audio, _ = librosa.load(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    audio_norm = audio.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec

# 音频处理函数
def pack_audio(audio_bytes, data, rate):
    if media_type == "ogg":
        audio_bytes = pack_ogg(audio_bytes, data, rate)
    elif media_type == "aac":
        audio_bytes = pack_aac(audio_bytes, data, rate)
    else:
        audio_bytes = pack_raw(audio_bytes, data, rate)
    return audio_bytes

def pack_ogg(audio_bytes, data, rate):
    with sf.SoundFile(audio_bytes, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return audio_bytes

def pack_raw(audio_bytes, data, rate):
    audio_bytes.write(data.tobytes())
    return audio_bytes

def pack_wav(audio_bytes, rate):
    if is_int32:
        data = np.frombuffer(audio_bytes.getvalue(), dtype=np.int32)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format='WAV', subtype='PCM_32')
    else:
        data = np.frombuffer(audio_bytes.getvalue(), dtype=np.int16)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format='WAV')
    return wav_bytes

def pack_aac(audio_bytes, data, rate):
    pcm = 's32le' if is_int32 else 's16le'
    bit_rate = '256k' if is_int32 else '128k'
    process = subprocess.Popen([
        'ffmpeg', '-f', pcm, '-ar', str(rate), '-ac', '1', '-i', 'pipe:0',
        '-c:a', 'aac', '-b:a', bit_rate, '-vn', '-f', 'adts', 'pipe:1'
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    audio_bytes.write(out)
    return audio_bytes

def read_clean_buffer(audio_bytes):
    audio_chunk = audio_bytes.getvalue()
    audio_bytes.truncate(0)
    audio_bytes.seek(0)
    return audio_bytes, audio_chunk

# 文本切分
def cut_text(text, punc):
    punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        if len(items) % 2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    return text

def only_punc(text):
    return not any(t.isalnum() or t.isalpha() for t in text)

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}


# TTS 推理函数
def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, top_k=15, top_p=0.6, temperature=0.6, speed=1, inp_refs=None, sample_steps=32, if_sr=False, spk="default", version=None):
    infer_sovits = speaker_list[spk].sovits
    vq_model = infer_sovits.vq_model
    hps = infer_sovits.hps
    
    # 如果提供了 version 参数，覆盖默认版本
    if version:
        hps.model.version = version
    
    infer_gpt = speaker_list[spk].gpt
    t2s_model = infer_gpt.t2s_model
    max_sec = infer_gpt.max_sec

    prompt_text = prompt_text.strip("\n")
    if prompt_text[-1] not in splits:
        prompt_text += "。" if prompt_language != "en" else "."
    prompt_language, text = prompt_language, text.strip("\n")
    dtype = torch.float16 if is_half else torch.float32
    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half else np.float32)
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)

        if hps.model.version != "v3":
            refers = []
            if inp_refs:
                for path in inp_refs:
                    try:
                        refer = get_spepc(hps, path).to(dtype).to(device)
                        refers.append(refer)
                    except Exception as e:
                        logger.error(e)
            if not refers:
                refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
        else:
            refer = get_spepc(hps, ref_wav_path).to(device).to(dtype)

    prompt_language = dict_language[prompt_language.lower()]
    text_language = dict_language[text_language.lower()]
    phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, hps.model.version)
    texts = text.split("\n")
    audio_bytes = BytesIO()

    for text in texts:
        if only_punc(text):
            continue
        audio_opt = []
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, hps.model.version)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

        if hps.model.version != "v3":
            audio = vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0),
                                    refers, speed=speed).detach().cpu().numpy()[0, 0]
        else:
            phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
            phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
            fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
            ref_audio, sr = torchaudio.load(ref_wav_path)
            ref_audio = ref_audio.to(device).float()
            if ref_audio.shape[0] == 2:
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            if sr != 24000:
                ref_audio = torchaudio.transforms.Resample(sr, 24000).to(device)(ref_audio)
            mel_fn = lambda x: torchaudio.transforms.MelSpectrogram(
                sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=100, f_min=0, f_max=None, center=False
            )(x)
            mel2 = mel_fn(ref_audio)
            mel2 = (mel2 - (-12)) / (2 - (-12)) * 2 - 1  # 简化的 norm_spec
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            if T_min > 468:
                mel2 = mel2[:, :, -468:]
                fea_ref = fea_ref[:, :, -468:]
                T_min = 468
            chunk_len = 934 - T_min
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
            cfm_resss = []
            idx = 0
            while True:
                fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
                if fea_todo_chunk.shape[-1] == 0:
                    break
                idx += chunk_len
                fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                cfm_res = vq_model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0)
                cfm_res = cfm_res[:, :, mel2.shape[2]:]
                mel2 = cfm_res[:, :, -T_min:]
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                cfm_resss.append(cfm_res)
            cmf_res = torch.cat(cfm_resss, 2)
            cmf_res = (cmf_res + 1) / 2 * (2 - (-12)) + (-12)  # 简化的 denorm_spec
            if bigvgan_model is None:
                init_bigvgan()
            with torch.inference_mode():
                wav_gen = bigvgan_model(cmf_res)
                audio = wav_gen[0][0].cpu().detach().numpy()

        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio /= max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        audio_opt = np.concatenate(audio_opt, 0)

        sr = hps.data.sampling_rate if hps.model.version != "v3" else 24000
        if if_sr and sr == 24000:
            audio_opt = torch.from_numpy(audio_opt).float().to(device)
            # 简化为无超分逻辑，需自行实现 audio_sr
            audio_opt = audio_opt.cpu().detach().numpy()
            sr = 48000

        if is_int32:
            audio_bytes = pack_audio(audio_bytes, (audio_opt * 2147483647).astype(np.int32), sr)
        else:
            audio_bytes = pack_audio(audio_bytes, (audio_opt * 32768).astype(np.int16), sr)
        if stream_mode == "normal":
            audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
            yield audio_chunk

    if stream_mode != "normal":
        if media_type == "wav":
            sr = 48000 if if_sr else 24000
            sr = hps.data.sampling_rate if hps.model.version != "v3" else sr
            audio_bytes = pack_wav(audio_bytes, sr)
        yield audio_bytes.getvalue()

# 接口处理函数
def handle_control(command):
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

def handle_change(path, text, language):
    if is_empty(path, text, language):
        return JSONResponse({"code": 400, "message": '缺少任意一项以下参数: "path", "text", "language"'}, status_code=400)
    if path:
        default_refer.path = path
    if text:
        default_refer.text = text
    if language:
        default_refer.language = language
    logger.info(f"当前默认参考音频路径: {default_refer.path}")
    logger.info(f"当前默认参考音频文本: {default_refer.text}")
    logger.info(f"当前默认参考音频语种: {default_refer.language}")
    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)

def handle(refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr):
    if not refer_wav_path or not prompt_text or not prompt_language:
        refer_wav_path, prompt_text, prompt_language = default_refer.path, default_refer.text, default_refer.language
        if not default_refer.is_ready():
            return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)
    if sample_steps not in [4, 8, 16, 32]:
        sample_steps = 32
    if cut_punc is None:
        text = cut_text(text, default_cut_punc)
    else:
        text = cut_text(text, cut_punc)
    return StreamingResponse(get_tts_wav(refer_wav_path, prompt_text, prompt_language, text, text_language, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr), media_type="audio/"+media_type)

def handle_ttsrole(text, role, text_language="auto", ref_audio_path=None, prompt_text=None, prompt_language=None, emotion=None, top_k=15, top_p=0.6, temperature=0.6, speed=1, inp_refs=None, sample_steps=32, if_sr=False, version=None, vits_weights_path=None, t2s_weights_path=None):
    if not text or not role:
        return JSONResponse({"code": 400, "message": "text and role are required"}, status_code=400)
    if role not in speaker_list:
        if not load_role_config(role, vits_weights_path, t2s_weights_path):
            return JSONResponse({"code": 400, "message": f"Role {role} not found"}, status_code=400)
    if not ref_audio_path:
        ref_audio_path, prompt_text_auto, prompt_lang_auto = select_ref_audio(role, text_language, emotion)
        if ref_audio_path:
            ref_audio_path, prompt_text, prompt_language = ref_audio_path, prompt_text_auto or prompt_text, prompt_lang_auto or prompt_language
        else:
            ref_audio_path, prompt_text, prompt_language = default_refer.path, default_refer.text, default_refer.language
            if not default_refer.is_ready():
                return JSONResponse({"code": 400, "message": "No reference audio provided and default not set"}, status_code=400)
    if sample_steps not in [4, 8, 16, 32]:
        sample_steps = 32
    text = cut_text(text, default_cut_punc)
    return StreamingResponse(get_tts_wav(ref_audio_path, prompt_text, prompt_language, text, text_language, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr, spk=role, version=version), media_type="audio/"+media_type)

# 初始化参数
dict_language = {
    "中文": "all_zh", "粤语": "all_yue", "英文": "en", "日文": "all_ja", "韩文": "all_ko",
    "中英混合": "zh", "粤英混合": "yue", "日英混合": "ja", "韩英混合": "ko", "多语种混合": "auto",
    "多语种混合(粤语)": "auto_yue", "all_zh": "all_zh", "all_yue": "all_yue", "en": "en",
    "all_ja": "all_ja", "all_ko": "all_ko", "zh": "zh", "yue": "yue", "ja": "ja", "ko": "ko",
    "auto": "auto", "auto_yue": "auto_yue"
}

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")
parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")
parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="使用半精度")
parser.add_argument("-sm", "--stream_mode", type=str, default="close", help="流式返回模式, close / normal")
parser.add_argument("-mt", "--media_type", type=str, default="wav", help="音频编码格式, wav / ogg / aac")
parser.add_argument("-st", "--sub_type", type=str, default="int16", help="音频数据类型, int16 / int32")
parser.add_argument("-cp", "--cut_punc", type=str, default="", help="文本切分符号设定")
parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")

args = parser.parse_args()
sovits_path = args.sovits_path
gpt_path = args.gpt_path
device = args.device
port = args.port
host = args.bind_addr
cnhubert_base_path = args.hubert_path
bert_path = args.bert_path
default_cut_punc = args.cut_punc

default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)
if not default_refer.path or not default_refer.text or not default_refer.language:
    default_refer.path, default_refer.text, default_refer.language = "", "", ""
    logger.info("未指定默认参考音频")
else:
    logger.info(f"默认参考音频路径: {default_refer.path}")
    logger.info(f"默认参考音频文本: {default_refer.text}")
    logger.info(f"默认参考音频语种: {default_refer.language}")

is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half
logger.info(f"半精: {is_half}")

stream_mode = "normal" if args.stream_mode.lower() in ["normal", "n"] else "close"
logger.info(f"流式返回: {'开启' if stream_mode == 'normal' else '关闭'}")

media_type = args.media_type.lower() if args.media_type.lower() in ["aac", "ogg"] else ("wav" if stream_mode == "close" else "ogg")
logger.info(f"编码格式: {media_type}")

is_int32 = args.sub_type.lower() == 'int32'
logger.info(f"数据类型: {'int32' if is_int32 else 'int16'}")

# 模型初始化
cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
ssl_model = cnhubert.get_model()
if is_half:
    bert_model = bert_model.half().to(device)
    ssl_model = ssl_model.half().to(device)
else:
    bert_model = bert_model.to(device)
    ssl_model = ssl_model.to(device)
change_gpt_sovits_weights(gpt_path=gpt_path, sovits_path=sovits_path)

# FastAPI 应用
app = FastAPI()

@app.post("/set_model")
async def set_model(request: Request):
    json_post_raw = await request.json()
    return change_gpt_sovits_weights(
        gpt_path=json_post_raw.get("gpt_model_path"),
        sovits_path=json_post_raw.get("sovits_model_path")
    )

@app.get("/set_model")
async def set_model(gpt_model_path: str = None, sovits_model_path: str = None):
    return change_gpt_sovits_weights(gpt_path=gpt_model_path, sovits_path=sovits_model_path)

@app.post("/control")
async def control(request: Request):
    json_post_raw = await request.json()
    return handle_control(json_post_raw.get("command"))

@app.get("/control")
async def control(command: str = None):
    return handle_control(command)

@app.post("/change_refer")
async def change_refer(request: Request):
    json_post_raw = await request.json()
    return handle_change(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_language")
    )

@app.get("/change_refer")
async def change_refer(refer_wav_path: str = None, prompt_text: str = None, prompt_language: str = None):
    return handle_change(refer_wav_path, prompt_text, prompt_language)

@app.post("/")
async def tts_endpoint(request: Request):
    json_post_raw = await request.json()
    return handle(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_language"),
        json_post_raw.get("text"),
        json_post_raw.get("text_language"),
        json_post_raw.get("cut_punc"),
        json_post_raw.get("top_k", 15),
        json_post_raw.get("top_p", 1.0),
        json_post_raw.get("temperature", 1.0),
        json_post_raw.get("speed", 1.0),
        json_post_raw.get("inp_refs", []),
        json_post_raw.get("sample_steps", 32),
        json_post_raw.get("if_sr", False)
    )

@app.get("/")
async def tts_endpoint(
    refer_wav_path: str = None, prompt_text: str = None, prompt_language: str = None, text: str = None, text_language: str = None,
    cut_punc: str = None, top_k: int = 15, top_p: float = 1.0, temperature: float = 1.0, speed: float = 1.0, inp_refs: list = Query(default=[]),
    sample_steps: int = 32, if_sr: bool = False
):
    return handle(refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr)

@app.post("/ttsrole")
async def ttsrole_endpoint(request: Request):
    json_post_raw = await request.json()
    return handle_ttsrole(
        json_post_raw.get("text"),
        json_post_raw.get("role"),
        json_post_raw.get("text_lang", "auto"),
        json_post_raw.get("ref_audio_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_lang"),
        json_post_raw.get("emotion"),
        json_post_raw.get("top_k", 15),
        json_post_raw.get("top_p", 0.6),
        json_post_raw.get("temperature", 0.6),
        json_post_raw.get("speed_factor", 1.0),
        json_post_raw.get("aux_ref_audio_paths", []),
        json_post_raw.get("sample_steps", 32),
        json_post_raw.get("if_sr", False),
        json_post_raw.get("version"),  # 支持动态切换版本
        json_post_raw.get("vits_weights_path"),  # 支持动态指定模型路径
        json_post_raw.get("t2s_weights_path")
    )

@app.get("/ttsrole")
async def ttsrole_endpoint(
    text: str, role: str, text_language: str = "auto", ref_audio_path: Optional[str] = None, prompt_text: Optional[str] = None,
    prompt_language: Optional[str] = None, emotion: Optional[str] = None, top_k: int = 15, top_p: float = 0.6,
    temperature: float = 0.6, speed: float = 1.0, inp_refs: list = Query(default=[]), sample_steps: int = 32, if_sr: bool = False, version: Optional[str] = None
):
    return handle_ttsrole(text, role, text_language, ref_audio_path, prompt_text, prompt_language, emotion, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr, version)

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, workers=1)
