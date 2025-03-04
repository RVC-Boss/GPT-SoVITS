"""
GPT-SoVITS API 实现

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
    "version": "v2",                   # str, 可选, 配置文件版本，覆盖 version
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
├── api_role.py                    # 本文件, API 主程序
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
   - python api_role.py -a 127.0.0.1 -p 9880
   - 检查启动日志确认设备
   
### 调用示例 (/ttsrole)
## 非流式调用，会一次性返回完整的音频数据，适用于需要完整音频文件的场景
import requests
url = "http://127.0.0.1:9880/ttsrole"
payload = {
    "text": "你好，这是一个测试",  # 要合成的文本
    "role": "role1",               # 角色名称，必填
    "emotion": "开心",              # 情感标签，可选
    "text_lang": "zh",             # 文本语言，可选，默认为 "zh"
    "media_type": "wav"            # 输出音频格式，默认 "wav"
}
response = requests.post(url, json=payload)
if response.status_code == 200:
    with open("output_non_stream.wav", "wb") as f:
        f.write(response.content)
    print("非流式音频已生成并保存为 output_non_stream.wav")
else:
    print(f"请求失败: {response.json()}")

## 流式调用，会分块返回音频数据，适用于实时播放或处理大文件的场景
import requests
url = "http://127.0.0.1:9880/ttsrole"
payload = {
    "text": "你好，这是一个测试",  # 要合成的文本
    "role": "role1",               # 角色名称，必填
    "emotion": "开心",              # 情感标签，可选
    "text_lang": "zh",             # 文本语言，可选，默认为 "zh"
    "media_type": "wav",           # 输出音频格式，默认 "wav"
    "streaming_mode": True         # 启用流式模式
}
with requests.post(url, json=payload, stream=True) as response:
    if response.status_code == 200:
        with open("output_stream.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # 确保 chunk 不为空
                    f.write(chunk)
        print("流式音频已生成并保存为 output_stream.wav")
    else:
        print(f"请求失败: {response.json()}")
"""

import os
import sys
import traceback
from typing import Generator, Optional, List, Dict
import random
import glob
from concurrent.futures import ThreadPoolExecutor
import asyncio

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names

# 尝试导入 PyTorch，检测显卡支持
try:
    import torch
    cuda_available = torch.cuda.is_available()
except ImportError:
    cuda_available = False
    print("缺少 PyTorch 依赖，默认使用 CPU")
except Exception as e:
    cuda_available = False
    print(f"检测显卡时出错: {str(e)}，默认使用 CPU")

i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT_SoVITS/configs/tts_infer.yaml"

default_device = "cuda" if cuda_available else "cpu"
print(f"默认设备设置为: {default_device}")

# 初始化 TTS 配置
tts_config = TTS_Config(config_path)
print(f"TTS_Config contents: {tts_config.__dict__}")
if hasattr(tts_config, 'device'):
    tts_config.device = default_device
tts_pipeline = TTS(tts_config)

# 创建线程池用于异步执行 TTS 任务
executor = ThreadPoolExecutor(max_workers=1)

APP = FastAPI()

class TTS_Request(BaseModel):
    text: str
    ref_audio_path: str
    prompt_lang: str
    text_lang: str = "auto"
    aux_ref_audio_paths: Optional[List[str]] = None
    prompt_text: Optional[str] = ""
    top_k: Optional[int] = 5
    top_p: Optional[float] = 1
    temperature: Optional[float] = 1
    text_split_method: Optional[str] = "cut5"
    batch_size: Optional[int] = 1
    batch_threshold: Optional[float] = 0.75
    split_bucket: Optional[bool] = True
    speed_factor: Optional[float] = 1.0
    fragment_interval: Optional[float] = 0.3
    seed: Optional[int] = -1
    media_type: Optional[str] = "wav"
    streaming_mode: Optional[bool] = False
    parallel_infer: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.35
    device: Optional[str] = None

class TTSRole_Request(BaseModel):
    text: str
    role: str
    text_lang: Optional[str] = "auto"
    ref_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[List[str]] = None
    prompt_lang: Optional[str] = None
    prompt_text: Optional[str] = None
    emotion: Optional[str] = None
    top_k: Optional[int] = 5
    top_p: Optional[float] = 1
    temperature: Optional[float] = 1
    text_split_method: Optional[str] = "cut5"
    batch_size: Optional[int] = 1
    batch_threshold: Optional[float] = 0.75
    split_bucket: Optional[bool] = True
    speed_factor: Optional[float] = 1.0
    fragment_interval: Optional[float] = 0.3
    seed: Optional[int] = -1
    media_type: Optional[str] = "wav"
    streaming_mode: Optional[bool] = False
    parallel_infer: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.35
    bert_base_path: Optional[str] = None
    cnhuhbert_base_path: Optional[str] = None
    device: Optional[str] = None
    is_half: Optional[bool] = None
    t2s_weights_path: Optional[str] = None
    version: Optional[str] = None
    vits_weights_path: Optional[str] = None
    t2s_model_path: Optional[str] = None
    vits_model_path: Optional[str] = None
    t2s_model_device: Optional[str] = None
    vits_model_device: Optional[str] = None

def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    io_buffer.seek(0)
    return io_buffer

def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    io_buffer.seek(0)
    return io_buffer

def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    sf.write(io_buffer, data, rate, format='wav')
    io_buffer.seek(0)
    return io_buffer

def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen([
        'ffmpeg', '-f', 's16le', '-ar', str(rate), '-ac', '1', '-i', 'pipe:0',
        '-c:a', 'aac', '-b:a', '192k', '-vn', '-f', 'adts', 'pipe:1'
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    io_buffer.seek(0)
    return io_buffer

def pack_audio(data: np.ndarray, rate: int, media_type: str) -> BytesIO:
    io_buffer = BytesIO()
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    return io_buffer

def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()

def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

def check_params(req: dict, is_ttsrole: bool = False):
    text = req.get("text")
    text_lang = req.get("text_lang", "auto")
    ref_audio_path = req.get("ref_audio_path")
    prompt_lang = req.get("prompt_lang")
    media_type = req.get("media_type", "wav")
    streaming_mode = req.get("streaming_mode", False)
    text_split_method = req.get("text_split_method", "cut5")

    if not text:
        return {"status": "error", "message": "text is required"}
    
    if is_ttsrole:
        role = req.get("role")
        if not role:
            return {"status": "error", "message": "role is required for /ttsrole"}
    else:
        if not ref_audio_path:
            return {"status": "error", "message": "ref_audio_path is required"}
        if not prompt_lang:
            return {"status": "error", "message": "prompt_lang is required"}
    
    languages = req.get("languages") or tts_config.languages
    if text_lang != "auto" and text_lang.lower() not in languages:
        return {"status": "error", "message": f"text_lang: {text_lang} is not supported"}
    if prompt_lang and prompt_lang.lower() not in languages:
        return {"status": "error", "message": f"prompt_lang: {prompt_lang} is not supported"}
    
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return {"status": "error", "message": f"media_type: {media_type} is not supported"}
    if media_type == "ogg" and not streaming_mode:
        return {"status": "error", "message": "ogg format is not supported in non-streaming mode"}
    if text_split_method not in cut_method_names:
        return {"status": "error", "message": f"text_split_method: {text_split_method} is not supported"}
    
    return None

def load_role_config(role: str, req: dict):
    role_dir = os.path.join(now_dir, "roles", role)
    if not os.path.exists(role_dir):
        return False
    
    if not any(req.get(k) for k in ["version", "bert_base_path", "cnhuhbert_base_path", "device", "is_half", "t2s_weights_path", "vits_weights_path"]):
        config_path_new = os.path.join(role_dir, "tts_infer.yaml")
        if os.path.exists(config_path_new):
            global tts_config, tts_pipeline
            tts_config = TTS_Config(config_path_new)
            if hasattr(tts_config, 'device'):
                tts_config.device = default_device
            tts_pipeline = TTS(tts_config)
    
    if not req.get("t2s_weights_path") and not req.get("t2s_model_path"):
        gpt_path = glob.glob(os.path.join(role_dir, "*.ckpt"))
        if gpt_path:
            tts_pipeline.init_t2s_weights(gpt_path[0])
    if not req.get("vits_weights_path") and not req.get("vits_model_path"):
        sovits_path = glob.glob(os.path.join(role_dir, "*.pth"))
        if sovits_path:
            tts_pipeline.init_vits_weights(sovits_path[0])
    
    return True

def select_ref_audio(role: str, text_lang: str, emotion: str = None):
    audio_base_dir = os.path.join(now_dir, "roles", role, "reference_audios")
    if not os.path.exists(audio_base_dir):
        return None, None, None
    
    if text_lang.lower() == "auto" and emotion:
        all_langs = [d for d in os.listdir(audio_base_dir) if os.path.isdir(os.path.join(audio_base_dir, d))]
        emotion_files = []
        for lang in all_langs:
            lang_dir = os.path.join(audio_base_dir, lang)
            emotion_files.extend(glob.glob(os.path.join(lang_dir, f"【{emotion}】*.*")))
        
        if emotion_files:
            audio_path = random.choice(emotion_files)
            txt_path = audio_path.rsplit(".", 1)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
            else:
                basename = os.path.basename(audio_path)
                start_idx = basename.find("】") + 1
                end_idx = basename.rfind(".")
                prompt_text = basename[start_idx:end_idx] if end_idx > start_idx else basename
            
            prompt_lang = os.path.basename(os.path.dirname(audio_path))
            return audio_path, prompt_text, prompt_lang
    
    lang_dir = os.path.join(audio_base_dir, text_lang.lower())
    all_langs = [d for d in os.listdir(audio_base_dir) if os.path.isdir(os.path.join(audio_base_dir, d))]
    
    def find_audio_in_dir(dir_path):
        if not os.path.exists(dir_path):
            return None, None
        audio_files = glob.glob(os.path.join(dir_path, "【*】*.*"))
        if not audio_files:
            audio_files = glob.glob(os.path.join(dir_path, "*.*"))
        if not audio_files:
            return None, None
        
        if emotion:
            emotion_files = [f for f in audio_files if f"【{emotion}】" in os.path.basename(f)]
            if emotion_files:
                audio_path = random.choice(emotion_files)
            else:
                audio_path = random.choice(audio_files)
        else:
            audio_path = random.choice(audio_files)
        
        txt_path = audio_path.rsplit(".", 1)[0] + ".txt"
        prompt_text = None
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        else:
            basename = os.path.basename(audio_path)
            start_idx = basename.find("】") + 1
            end_idx = basename.rfind(".")
            if start_idx > 0 and end_idx > start_idx:
                prompt_text = basename[start_idx:end_idx]
            else:
                prompt_text = basename[:end_idx] if end_idx > 0 else basename
        
        return audio_path, prompt_text
    
    audio_path, prompt_text = find_audio_in_dir(lang_dir)
    if audio_path:
        return audio_path, prompt_text, text_lang.lower()
    
    for lang in all_langs:
        if lang != text_lang.lower():
            audio_path, prompt_text = find_audio_in_dir(os.path.join(audio_base_dir, lang))
            if audio_path:
                return audio_path, prompt_text, lang
    
    return None, None, None

def set_pipeline_device(pipeline: TTS, device: str):
    """将 TTS 管道中的所有模型和相关组件迁移到指定设备，仅在设备变化时执行"""
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print(f"警告: CUDA 不可用，强制使用 CPU")
        device = "cpu"
    
    target_device = torch.device(device)
    
    # 检查当前设备是否需要切换
    current_device = None
    if hasattr(pipeline, 't2s_model') and pipeline.t2s_model is not None:
        current_device = next(pipeline.t2s_model.parameters()).device
    elif hasattr(pipeline, 'vits_model') and pipeline.vits_model is not None:
        current_device = next(pipeline.vits_model.parameters()).device
    
    if current_device == target_device:
        print(f"设备已是 {device}，无需切换")
        return
    
    # 更新配置中的设备
    if hasattr(pipeline, 'configs') and hasattr(pipeline.configs, 'device'):
        pipeline.configs.device = device
    
    # 迁移所有可能的模型到指定设备
    for attr in ['t2s_model', 'vits_model']:
        if hasattr(pipeline, attr) and getattr(pipeline, attr) is not None:
            getattr(pipeline, attr).to(target_device)
    
    for attr in dir(pipeline):
        if attr.endswith('_model') and getattr(pipeline, attr) is not None:
            try:
                getattr(pipeline, attr).to(target_device)
                print(f"迁移 {attr} 到 {device}")
            except AttributeError:
                pass
    
    # 清理 GPU 缓存
    if torch.cuda.is_available() and not device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    print(f"TTS 管道设备已设置为: {device}")

def run_tts_pipeline(req):
    """在线程池中运行 TTS 任务"""
    return tts_pipeline.run(req)

async def tts_handle(req: dict, is_ttsrole: bool = False):
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    if "text_lang" not in req:
        req["text_lang"] = "auto"

    check_res = check_params(req, is_ttsrole)
    if check_res is not None:
        return JSONResponse(status_code=400, content=check_res)
    
    # 如果请求中指定了 device，则覆盖所有与设备相关的参数并更新管道设备
    if "device" in req and req["device"] is not None:
        device = req["device"]
        req["t2s_model_device"] = device
        req["vits_model_device"] = device
        if hasattr(tts_config, 'device'):
            tts_config.device = device
        set_pipeline_device(tts_pipeline, device)
    
    if is_ttsrole:
        role_exists = load_role_config(req["role"], req)
        
        for key in ["bert_base_path", "cnhuhbert_base_path", "device", "is_half", "t2s_weights_path", "version", "vits_weights_path"]:
            if req.get(key) is not None:
                setattr(tts_config, key, req[key])
        
        if req.get("t2s_model_path"):
            tts_config.t2s_weights_path = req["t2s_model_path"]
            tts_pipeline.init_t2s_weights(req["t2s_model_path"])
        if req.get("vits_model_path"):
            tts_config.vits_weights_path = req["vits_model_path"]
            tts_pipeline.init_vits_weights(req["vits_model_path"])
        
        if not req.get("ref_audio_path"):
            ref_audio_path, prompt_text, prompt_lang = select_ref_audio(req["role"], req["text_lang"], req.get("emotion"))
            if ref_audio_path:
                req["ref_audio_path"] = ref_audio_path
                req["prompt_text"] = prompt_text or ""
                req["prompt_lang"] = prompt_lang or req["text_lang"]
            elif not role_exists:
                return JSONResponse(status_code=400, content={"status": "error", "message": "Role directory not found and no suitable reference audio provided"})
        else:
            ref_audio_path = req["ref_audio_path"]
            txt_path = ref_audio_path.rsplit(".", 1)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    req["prompt_text"] = f.read().strip()
            else:
                basename = os.path.basename(ref_audio_path)
                if "【" in basename and "】" in basename:
                    start_idx = basename.find("】") + 1
                    end_idx = basename.rfind(".")
                    if start_idx > 0 and end_idx > start_idx:
                        req["prompt_text"] = basename[start_idx:end_idx]
                    else:
                        req["prompt_text"] = basename[:end_idx] if end_idx > 0 else basename
                else:
                    end_idx = basename.rfind(".")
                    req["prompt_text"] = basename[:end_idx] if end_idx > 0 else basename
    
    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        print(f"当前请求设备: {req.get('device')}")
        if hasattr(tts_pipeline, 't2s_model') and tts_pipeline.t2s_model is not None:
            print(f"t2s_model 设备: {next(tts_pipeline.t2s_model.parameters()).device}")
        if hasattr(tts_pipeline, 'vits_model') and tts_pipeline.vits_model is not None:
            print(f"vits_model 设备: {next(tts_pipeline.vits_model.parameters()).device}")
        
        # 异步执行 TTS 任务
        loop = asyncio.get_event_loop()
        tts_generator = await loop.run_in_executor(executor, run_tts_pipeline, req)
        
        if streaming_mode:
            def streaming_generator():
                if media_type == "wav":
                    yield wave_header_chunk()
                    stream_type = "raw"
                else:
                    stream_type = media_type
                for sr, chunk in tts_generator:
                    buf = pack_audio(chunk, sr, stream_type)
                    yield buf.getvalue()
            return StreamingResponse(streaming_generator(), media_type=f"audio/{media_type}")
        else:
            sr, audio_data = next(tts_generator)
            buf = pack_audio(audio_data, sr, media_type)
            return Response(buf.getvalue(), media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": "tts failed", "exception": str(e)})

@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "command is required"})
    handle_control(command)

@APP.get("/tts")
async def tts_get_endpoint(
    text: str,
    ref_audio_path: str,
    prompt_lang: str,
    text_lang: str = "auto",
    aux_ref_audio_paths: Optional[List[str]] = None,
    prompt_text: Optional[str] = "",
    top_k: Optional[int] = 5,
    top_p: Optional[float] = 1,
    temperature: Optional[float] = 1,
    text_split_method: Optional[str] = "cut0",
    batch_size: Optional[int] = 1,
    batch_threshold: Optional[float] = 0.75,
    split_bucket: Optional[bool] = True,
    speed_factor: Optional[float] = 1.0,
    fragment_interval: Optional[float] = 0.3,
    seed: Optional[int] = -1,
    media_type: Optional[str] = "wav",
    streaming_mode: Optional[bool] = False,
    parallel_infer: Optional[bool] = True,
    repetition_penalty: Optional[float] = 1.35,
    device: Optional[str] = None
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_lang": prompt_lang.lower(),
        "prompt_text": prompt_text,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "split_bucket": split_bucket,
        "speed_factor": speed_factor,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "device": device
    }
    return await tts_handle(req)

@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict(exclude_unset=True)
    if "text_lang" in req:
        req["text_lang"] = req["text_lang"].lower()
    if "prompt_lang" in req:
        req["prompt_lang"] = req["prompt_lang"].lower()
    return await tts_handle(req)

@APP.get("/ttsrole")
async def ttsrole_get_endpoint(
    text: str,
    role: str,
    text_lang: str = "auto",
    ref_audio_path: Optional[str] = None,
    aux_ref_audio_paths: Optional[List[str]] = None,
    prompt_lang: Optional[str] = None,
    prompt_text: Optional[str] = None,
    emotion: Optional[str] = None,
    top_k: Optional[int] = 5,
    top_p: Optional[float] = 1,
    temperature: Optional[float] = 1,
    text_split_method: Optional[str] = "cut5",
    batch_size: Optional[int] = 1,
    batch_threshold: Optional[float] = 0.75,
    split_bucket: Optional[bool] = True,
    speed_factor: Optional[float] = 1.0,
    fragment_interval: Optional[float] = 0.3,
    seed: Optional[int] = -1,
    media_type: Optional[str] = "wav",
    streaming_mode: Optional[bool] = False,
    parallel_infer: Optional[bool] = True,
    repetition_penalty: Optional[float] = 1.35,
    bert_base_path: Optional[str] = None,
    cnhuhbert_base_path: Optional[str] = None,
    device: Optional[str] = None,
    is_half: Optional[bool] = None,
    t2s_weights_path: Optional[str] = None,
    version: Optional[str] = None,
    vits_weights_path: Optional[str] = None,
    t2s_model_path: Optional[str] = None,
    vits_model_path: Optional[str] = None,
    t2s_model_device: Optional[str] = None,
    vits_model_device: Optional[str] = None
):
    req = {
        "text": text,
        "role": role,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_lang": prompt_lang.lower() if prompt_lang else None,
        "prompt_text": prompt_text,
        "emotion": emotion,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "split_bucket": split_bucket,
        "speed_factor": speed_factor,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "bert_base_path": bert_base_path,
        "cnhuhbert_base_path": cnhuhbert_base_path,
        "device": device,
        "is_half": is_half,
        "t2s_weights_path": t2s_weights_path,
        "version": version,
        "vits_weights_path": vits_weights_path,
        "t2s_model_path": t2s_model_path,
        "vits_model_path": vits_model_path,
        "t2s_model_device": t2s_model_device,
        "vits_model_device": vits_model_device
    }
    return await tts_handle(req, is_ttsrole=True)

@APP.post("/ttsrole")
async def ttsrole_post_endpoint(request: TTSRole_Request):
    req = request.dict(exclude_unset=True)
    if "text_lang" in req:
        req["text_lang"] = req["text_lang"].lower()
    if "prompt_lang" in req:
        req["prompt_lang"] = req["prompt_lang"].lower()
    return await tts_handle(req, is_ttsrole=True)

@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if not weights_path:
            return JSONResponse(status_code=400, content={"status": "error", "message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
        tts_config.t2s_weights_path = weights_path
        return JSONResponse(status_code=200, content={"status": "success", "message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"change gpt weight failed", "exception": str(e)})

@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if not weights_path:
            return JSONResponse(status_code=400, content={"status": "error", "message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
        tts_config.vits_weights_path = weights_path
        return JSONResponse(status_code=200, content={"status": "success", "message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"change sovits weight failed", "exception": str(e)})

@APP.get("/set_refer_audio")
async def set_refer_audio(refer_audio_path: str = None):
    try:
        if not refer_audio_path:
            return JSONResponse(status_code=400, content={"status": "error", "message": "refer audio path is required"})
        tts_pipeline.set_ref_audio(refer_audio_path)
        return JSONResponse(status_code=200, content={"status": "success", "message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"set refer audio failed", "exception": str(e)})

if __name__ == "__main__":
    try:
        if host == 'None':  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)