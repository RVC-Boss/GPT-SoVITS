"""
# GPT-SoVITS API 使用说明
# ======================
# 
# 本API服务提供与OpenAI TTS API兼容的接口，基于GPT-SoVITS进行语音合成，不支持v1，v2版本模型
#
# 主要接口:
# 1. 语音合成: POST /v1/audio/speech
#    示例请求:
#    ```
#    curl -X POST http://localhost:15000/v1/audio/speech \
#      -H "Content-Type: application/json" \
#      -d '{
#        "model": "tts-1",
#        "input": "你好，这是一段测试文本",
#        "voice": "训练的项目模型名称",
#        "response_format": "mp3",
#        "speed": 1.0
#      }'
#    ```
#
# 2. 查看可用模型: GET /v1/voices
#    示例请求:
#    ```
#    curl http://localhost:15000/v1/voices
#    ```
#
# 3. 上传参考音频: POST /v1/voices/{voice_id}/reference
#    示例请求:
#    ```
#    curl -X POST http://localhost:15000/v1/voices/your_model_name/reference \
#      -F "file=@/path/to/reference.wav" \
#      -F "description=参考音频描述"
#    ```
#
# 4. 获取模型参考音频列表: GET /v1/voices/{voice_id}/references
#    示例请求:
#    ```
#    curl http://localhost:15000/v1/voices/your_model_name/references
#    ```
#
# 5. 健康检查: GET /health
#    示例请求:
#    ```
#    curl http://localhost:15000/health
#    ```
#
# 更多接口详情请查看API文档: http://localhost:15000/docs
#
"""
import os
import sys
import re
import json
import tempfile
from typing import List, Optional, Dict, Tuple
import uuid
import time
import base64
import random

import torch
import librosa
import numpy as np
import torchaudio
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

# 在开头导入模块后添加
from model_manager import model_manager

# 将当前目录加入系统路径
current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "GPT_SoVITS"))

# 导入GPT-SoVITS相关模块
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3, Generator
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.module.mel_processing import spectrogram_torch, mel_spectrogram_torch
from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 配置日志
import logging

# 创建日志格式化器
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# 配置根日志记录器
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

# 配置应用日志记录器
logger = logging.getLogger("gpt-sovits-api")
logger.setLevel(logging.INFO)
# 防止日志重复
logger.propagate = False
logger.addHandler(console_handler)

# 配置uvicorn访问日志
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.propagate = False
uvicorn_logger.addHandler(console_handler)

# 配置uvicorn错误日志
uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_error_logger.propagate = False
uvicorn_error_logger.addHandler(console_handler)

# 配置fastapi访问日志
fastapi_logger = logging.getLogger("fastapi")
fastapi_logger.propagate = False
fastapi_logger.addHandler(console_handler)

# 全局变量
device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = torch.cuda.is_available()
AUDIO_CACHE = {}  # 用于缓存音频，避免重复计算

# 配置类
class Config:
    def __init__(self):
        self.sovits_path = os.environ.get("SOVITS_PATH", "GPT_SoVITS/pretrained_models/s2Gv3.pth")
        self.gpt_path = os.environ.get("GPT_PATH", "GPT_SoVITS/pretrained_models/s1v3.ckpt")
        self.cnhubert_base_path = os.environ.get("CNHUBERT_PATH", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
        self.bert_path = os.environ.get("BERT_PATH", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
        self.port = int(os.environ.get("PORT", 15000))
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.stream_mode = os.environ.get("STREAM_MODE", "keepalive") # close, normal, keepalive
        self.audio_type = os.environ.get("AUDIO_TYPE", "wav") # wav, ogg, aac
        self.default_top_k = 20
        self.default_top_p = 0.6
        self.default_temperature = 0.6
        self.default_speed = 1.0
        self.api_keys = os.environ.get("API_KEYS", "").split(",")

config = Config()

# 初始化模型
ssl_model = None
bert_model = None
tokenizer = None
vq_model = None
hps = None
t2s_model = None
bigvgan_model = None
hifigan_model = None
hz = 50
max_sec = 30
model_version = "v3"  # 默认版本
version = "v2"  # 默认版本
v3v4set = {"v3", "v4"}
if_lora_v3 = False

# OpenAI API 模型
class TTSRequest(BaseModel):
    model: str = Field(default="tts-1")  # 模型名称
    input: str  # 需要合成的文本
    voice: str  # 声音ID，实际是参考音频路径
    voice_text: Optional[str] = None  # 参考音频的文本
    voice_language: Optional[str] = "auto"  # 参考音频的语言，默认auto
    text_language: Optional[str] = "auto"  # 合成文本的语言，默认auto
    response_format: Optional[str] = "mp3"  # 响应格式，支持 mp3, wav, ogg
    speed: Optional[float] = 1.0  # 语速
    temperature: Optional[float] = 0.6  # 温度
    top_p: Optional[float] = 0.6  # top_p
    top_k: Optional[int] = 20  # top_k
    sample_steps: Optional[int] = 32  # 采样步数
    cut_method: Optional[str] = "凑四句一切"  # 切分方式，可选：凑四句一切/凑50字一切/按中文句号。切/按英文句号.切/按标点符号切/不切

class TTSResponse(BaseModel):
    model: str
    created: int
    audio: Optional[str] = None  # Base64编码的音频数据

app = FastAPI(title="GPT-SoVITS TTS API", description="OpenAI compatible TTS API using GPT-SoVITS")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 辅助类，用于将字典转为属性
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

# 初始化需要的全局变量和模型
def init_models():
    global ssl_model, bert_model, tokenizer, model_version, version, hps, vq_model, t2s_model, if_lora_v3
    
    # 初始化SSL模型 - 直接创建CNHubert实例并传入正确的路径
    hubert_base_path = os.path.join(current_dir, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")
    ssl_model = cnhubert.CNHubert(hubert_base_path)
    ssl_model.eval()
    if is_half:
        ssl_model = ssl_model.half().to(device)
    else:
        ssl_model = ssl_model.to(device)
    
    # 初始化BERT模型
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(config.bert_path)
    if is_half:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)
    
    # 加载SoVITS模型
    load_sovits_model(config.sovits_path)
    
    # 加载GPT模型
    load_gpt_model(config.gpt_path)

def init_bigvgan():
    """初始化BigVGAN模型"""
    global bigvgan_model, hifigan_model
    try:
        from GPT_SoVITS.BigVGAN import bigvgan
        
        logger.info("开始加载BigVGAN模型")
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            f"{current_dir}/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x",
            use_cuda_kernel=False,
        )
        
        if bigvgan_model is None:
            raise RuntimeError("BigVGAN模型加载失败")
            
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval()
        
        # 清理hifigan
        if hifigan_model is not None:
            hifigan_model = hifigan_model.cpu()
            hifigan_model = None
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        # 移动到正确的设备
        if is_half:
            bigvgan_model = bigvgan_model.half().to(device)
        else:
            bigvgan_model = bigvgan_model.to(device)
            
        logger.info("BigVGAN模型加载完成")
        return bigvgan_model
    except Exception as e:
        logger.error(f"加载BigVGAN模型失败: {str(e)}")
        logger.exception(e)
        raise

def init_hifigan():
    """初始化HiFiGAN模型"""
    global hifigan_model, bigvgan_model
    try:
        logger.info("开始加载HiFiGAN模型")
        hifigan_model = Generator(
            initial_channel=100,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 6, 2, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[20, 12, 4, 4, 4],
            gin_channels=0, is_bias=True
        )
        
        if hifigan_model is None:
            raise RuntimeError("HiFiGAN模型初始化失败")
            
        hifigan_model.eval()
        hifigan_model.remove_weight_norm()
        
        # 加载权重
        state_dict_path = f"{current_dir}/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth"
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"HiFiGAN权重文件不存在: {state_dict_path}")
            
        state_dict_g = torch.load(state_dict_path, map_location="cpu")
        logger.info("加载vocoder权重")
        load_result = hifigan_model.load_state_dict(state_dict_g)
        logger.info(f"HiFiGAN权重加载结果: {load_result}")
        
        # 清理bigvgan
        if bigvgan_model is not None:
            bigvgan_model = bigvgan_model.cpu()
            bigvgan_model = None
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        # 移动到正确的设备
        if is_half:
            hifigan_model = hifigan_model.half().to(device)
        else:
            hifigan_model = hifigan_model.to(device)
            
        logger.info("HiFiGAN模型加载完成")
        return hifigan_model
    except Exception as e:
        logger.error(f"加载HiFiGAN模型失败: {str(e)}")
        logger.exception(e)
        raise

# 获取音频特征
def get_spepc(hps, filename):
    audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

# 清理文本
def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

# 获取BERT特征
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

# 获取BERT特征（针对不同语言）
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

# 获取音素和BERT特征
splits = {
    "，", "。", "？", "！", ",", ".", "?", "!", "~", ":",
    "：", "—", "…"
}

def get_phones_and_bert(text, language, version, final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "all_zh":
            if re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                from GPT_SoVITS.text import chinese
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext, "zh", version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
            formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
            from GPT_SoVITS.text import chinese
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
                    # 因无法区别中日韩文汉字,以用户输入为准
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
        norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    dtype = torch.float16 if is_half else torch.float32
    return phones, bert.to(dtype), norm_text

# Mel谱处理函数
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

# 重采样函数
resample_transform_dict = {}

def resample(audio_tensor, sr0, sr1):
    global resample_transform_dict
    key = f"{sr0}-{sr1}"
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

# 音频处理函数
def process_audio(data, sr, format):
    """处理音频数据，转换为指定格式"""
    if format == "mp3":
        import io
        try:
            import soundfile as sf
            import pydub
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("请安装 soundfile 和 pydub 库以支持 mp3 格式")
        
        # 保存为临时 WAV 文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, data, sr, format="WAV")
        
        # 转换为 MP3
        audio = AudioSegment.from_wav(tmp_wav.name)
        mp3_io = io.BytesIO()
        audio.export(mp3_io, format="mp3", bitrate="256k")
        os.unlink(tmp_wav.name)  # 删除临时文件
        
        return mp3_io.getvalue()
    
    elif format == "wav":
        import io
        import wave
        import struct
        
        # 将float转换为16位整数
        pcm_data = (data * 32767).astype(np.int16).tobytes()
        
        # 创建WAV格式
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sr)
            wav_file.writeframes(pcm_data)
        
        return wav_io.getvalue()
    
    elif format == "ogg":
        import io
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("请安装 soundfile 库以支持 ogg 格式")
        
        ogg_io = io.BytesIO()
        sf.write(ogg_io, data, sr, format="OGG")
        
        return ogg_io.getvalue()
    
    else:
        raise ValueError(f"不支持的音频格式: {format}")

# 完全重写模型缓存和预加载逻辑
# 使用单例模式存储已加载的模型，而不是每次重新加载
class ModelLoader:
    """模型加载器，用于管理和缓存已加载的模型"""
    
    def __init__(self):
        self.model_cache = {}
        self.is_loading = False
        self.default_model_loaded = False
    
    def load_model(self, voice_model):
        """加载指定模型到缓存"""
        global t2s_model, vq_model, hps, version, model_version, if_lora_v3
        
        # 如果模型已经在缓存中，直接使用
        if voice_model in self.model_cache:
            logger.info(f"使用缓存模型: {voice_model}")
            cached_model = self.model_cache[voice_model]
            t2s_model = cached_model["t2s_model"]
            vq_model = cached_model["vq_model"]
            hps = cached_model["hps"]
            version = cached_model["version"]
            model_version = cached_model["model_version"]
            if_lora_v3 = cached_model["if_lora_v3"]
            
            # 确保所有模型都在正确的设备上
            if is_half:
                t2s_model = t2s_model.half().to(device)
                vq_model = vq_model.half().to(device)
            else:
                t2s_model = t2s_model.to(device)
                vq_model = vq_model.to(device)
                
            return True
        
        # 如果模型不在缓存中，加载新模型
        logger.info(f"加载新模型: {voice_model}")
        gpt_path, sovits_path = model_manager.get_model_paths(voice_model)
        if not gpt_path or not sovits_path:
            logger.error(f"未找到模型 {voice_model} 的路径")
            return False
        
        try:
            self.is_loading = True
            # 加载新模型
            load_gpt_model(gpt_path)
            load_sovits_model(sovits_path)
            
            # 将加载的模型保存到缓存
            self.model_cache[voice_model] = {
                "t2s_model": t2s_model.cpu(),  # 存入缓存前先移到CPU
                "vq_model": vq_model.cpu(),    # 存入缓存前先移到CPU
                "hps": hps,
                "version": version,
                "model_version": model_version,
                "if_lora_v3": if_lora_v3,
                "gpt_path": gpt_path,
                "sovits_path": sovits_path
            }
            self.is_loading = False
            return True
        except Exception as e:
            self.is_loading = False
            logger.error(f"加载模型 {voice_model} 失败: {str(e)}")
            logger.exception(e)
            return False
    
    def get_default_model(self):
        """获取默认模型"""
        if not self.default_model_loaded:
            # 加载默认模型
            self.load_model(next(iter(model_manager.model_mapping)))
            self.default_model_loaded = True
        
        # 返回第一个模型
        return next(iter(self.model_cache.values()))
    
    def preload_models(self, model_names=None):
        """预加载多个模型"""
        if model_names is None:
            # 如果没有指定模型名称，加载所有模型
            model_names = [voice["name"] for voice in model_manager.get_all_voices()]
        
        for model_name in model_names:
            if model_name not in self.model_cache:
                self.load_model(model_name)
    
    def get_model_info(self, model_name):
        """获取模型信息"""
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        return None

# 创建模型加载器实例
model_loader = ModelLoader()

# 添加一个修复设备不匹配的函数
def ensure_model_on_device(model, target_device, half_precision=False):
    """
    确保模型的所有组件都在同一设备上
    
    Args:
        model: PyTorch模型
        target_device: 目标设备
        half_precision: 是否使用半精度
    
    Returns:
        确保在目标设备上的模型
    """
    if model is None:
        logger.warning("模型为None，跳过设备转换")
        return model
        
    if not isinstance(model, torch.nn.Module):
        return model
    
    try:
        # 获取当前设备
        try:
            current_device = next(model.parameters()).device
        except StopIteration:
            if half_precision:
                model = model.half()
            return model.to(target_device)
            
        # 如果已经在目标设备上且精度正确，直接返回
        if str(current_device) == str(target_device):
            if not half_precision or (half_precision and next(model.parameters()).dtype == torch.float16):
                return model
        
        # 先将模型移到CPU，再移到目标设备
        model = model.cpu()
        if half_precision:
            model = model.half()
        model = model.to(target_device)
        
        return model
    except Exception as e:
        logger.error(f"移动模型到设备{target_device}时出错: {str(e)}")
        logger.exception(e)
        return model

# 添加vocoder模型管理函数
def ensure_vocoder_loaded():
    """确保vocoder模型正确加载"""
    global bigvgan_model, hifigan_model, model_version
    
    try:
        if model_version == "v3":
            if bigvgan_model is None:
                logger.info("加载BigVGAN模型")
                init_bigvgan()
            return bigvgan_model
        else:  # v4
            if hifigan_model is None:
                logger.info("加载HiFiGAN模型")
                init_hifigan()
            return hifigan_model
    except Exception as e:
        logger.error(f"加载vocoder模型失败: {str(e)}")
        logger.exception(e)
        raise RuntimeError("Vocoder模型加载失败") from e

# 修改合成语音的主函数，解决长文本和空文本问题
def synthesize_speech(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    speed=1.0,
    sample_steps=8,
    voice_model=None,  # 新增voice_model参数
):
    """生成语音"""
    global AUDIO_CACHE, t2s_model, vq_model, hps, version, model_version, if_lora_v3, ssl_model, bigvgan_model, hifigan_model
    
    # 验证参数
    if not ref_wav_path:
        raise ValueError("缺少参考音频路径")
    
    if not text or not text.strip():
        raise ValueError("缺少需要合成的文本")
    
    # 检查模型版本是否支持
    if model_version in ["v1", "v2"]:
        error_msg = f"不支持的模型版本: {model_version}，目前只支持v3和v4版本的模型"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 如果指定了voice_model，加载对应的模型
    if voice_model and voice_model in model_manager.model_mapping:
        model_loader.load_model(voice_model)
    
    # 确保所有模型都在相同的设备上
    try:
        # 确保SSL模型在正确设备上
        ssl_model = ensure_model_on_device(ssl_model, device, is_half)
        
        # 确保SSL模型内部组件在正确设备上
        if hasattr(ssl_model, 'model'):
            ssl_model.model = ensure_model_on_device(ssl_model.model, device, is_half)
        
        # 确保VQ模型在正确设备上
        vq_model = ensure_model_on_device(vq_model, device, is_half)
        
        # 确保VQ模型内部组件在正确设备上
        for attr_name in dir(vq_model):
            if attr_name.startswith('__'):  # 跳过内置属性
                continue
            attr = getattr(vq_model, attr_name)
            if isinstance(attr, torch.nn.Module):
                setattr(vq_model, attr_name, ensure_model_on_device(attr, device, is_half))
        
        # 特别处理ssl_proj
        if hasattr(vq_model, 'ssl_proj'):
            try:
                # 如果不在目标设备上，移到目标设备
                if str(next(vq_model.ssl_proj.parameters()).device) != str(device):
                    vq_model.ssl_proj = vq_model.ssl_proj.to(device)
            except Exception as e:
                logger.error(f"处理vq_model.ssl_proj时出错: {str(e)}")
        
        # 确保T2S模型在正确设备上
        t2s_model = ensure_model_on_device(t2s_model, device, is_half)
        
    except Exception as e:
        logger.error(f"确保模型在设备上时出错: {str(e)}")
        logger.exception(e)
    
    # 创建一个缓存键
    cache_key = f"{ref_wav_path}_{prompt_text}_{prompt_language}_{text}_{text_language}_{top_k}_{top_p}_{temperature}_{speed}_{sample_steps}"
    cache_key += f"_{voice_model}" if voice_model else ""
    
    if cache_key in AUDIO_CACHE:
        logger.info("使用缓存的音频结果")
        return AUDIO_CACHE[cache_key]
    
    # 添加静音
    one_sec_silence = np.zeros(int(hps.data.sampling_rate * 0.5), dtype=np.float16 if is_half else np.float32)
    
    # 处理参考音频
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise ValueError("参考音频应该在3~10秒范围内")
        
        wav16k = torch.from_numpy(wav16k)
        if is_half:
            wav16k = wav16k.half().to(device)
        else:
            wav16k = wav16k.to(device)
        
        # 添加静音
        zero_wav = torch.zeros(
            int(hps.data.sampling_rate * 0.3),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)
        wav16k = torch.cat([wav16k, zero_wav])
        
        # 提取SSL内容 - 确保模型和输入在相同设备上
        try:
            # 再次检查SSL模型设备，确保所有组件都在同一设备上
            if hasattr(ssl_model, 'model'):
                # 确保模型的weight在正确设备上
                for param_name, param in ssl_model.model.named_parameters():
                    if param.device != device:
                        param.data = param.data.to(device)
            
            # 确保vq_model.ssl_proj在正确设备上
            if hasattr(vq_model, 'ssl_proj'):
                for param_name, param in vq_model.ssl_proj.named_parameters():
                    if param.device != device:
                        param.data = param.data.to(device)
            
            # 进行推理
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)
        except RuntimeError as e:
            # 如果出现设备不匹配错误，尝试更强力的方法
            if "Input type" in str(e) and "weight type" in str(e):
                logger.error(f"设备不匹配错误: {str(e)}")
                # 尝试通过reset_parameters方法修复
                if hasattr(vq_model, 'ssl_proj') and hasattr(vq_model.ssl_proj, 'reset_parameters'):
                    logger.info("尝试重置ssl_proj参数来修复设备不匹配")
                    # 备份权重
                    weight_backup = vq_model.ssl_proj.weight.data.clone()
                    bias_backup = vq_model.ssl_proj.bias.data.clone() if hasattr(vq_model.ssl_proj, 'bias') and vq_model.ssl_proj.bias is not None else None
                    
                    # 重置参数
                    vq_model.ssl_proj.reset_parameters()
                    
                    # 将备份的权重移到正确设备上并恢复
                    vq_model.ssl_proj.weight.data = weight_backup.to(device)
                    if bias_backup is not None:
                        vq_model.ssl_proj.bias.data = bias_backup.to(device)
                    
                    # 再次尝试推理
                    ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
                    codes = vq_model.extract_latent(ssl_content)
                    prompt_semantic = codes[0, 0]
                    prompt = prompt_semantic.unsqueeze(0).to(device)
                else:
                    # 如果无法修复，重新抛出异常
                    raise
            else:
                # 其他运行时错误，重新抛出
                raise
        except Exception as e:
            logger.error(f"SSL处理错误: {str(e)}")
            logger.exception(e)
            raise
    
    # 处理文本
    phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
    phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
    
    # 合并特征
    bert = torch.cat([bert1, bert2], 1)
    all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
    bert = bert.to(device).unsqueeze(0)
    all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
    
    # 生成语义
    with torch.no_grad():
        # 使用和inference_webui.py相同的推理方式
        pred_semantic, idx = t2s_model.model.infer_panel(
            all_phoneme_ids,
            all_phoneme_len,
            prompt,
            bert,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=hz * max_sec,
        )
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
    
    # 解码生成音频
    refer = get_spepc(hps, ref_wav_path).to(device).to(torch.float16 if is_half else torch.float32)
    phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
    phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
    
    fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
    ref_audio, sr = torchaudio.load(ref_wav_path)
    ref_audio = ref_audio.to(device).float()
    if ref_audio.shape[0] == 2:
        ref_audio = ref_audio.mean(0).unsqueeze(0)
    
    tgt_sr = 24000 if model_version == "v3" else 32000
    if sr != tgt_sr:
        ref_audio = resample(ref_audio, sr, tgt_sr)
    
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
    mel2 = mel2.to(torch.float16 if is_half else torch.float32)
    fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
    
    cfm_resss = []
    idx = 0
    while True:
        fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
        if fea_todo_chunk.shape[-1] == 0:
            break
        idx += chunk_len
        fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
        cfm_res = vq_model.cfm.inference(
            fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
        )
        cfm_res = cfm_res[:, :, mel2.shape[2]:]
        mel2 = cfm_res[:, :, -T_min:]
        fea_ref = fea_todo_chunk[:, :, -T_min:]
        cfm_resss.append(cfm_res)
    
    cfm_res = torch.cat(cfm_resss, 2)
    cfm_res = denorm_spec(cfm_res)
    
    # 使用vocoder生成波形
    try:
        # 确保vocoder模型已加载并在正确设备上
        vocoder_model = ensure_vocoder_loaded()
        if vocoder_model is None:
            raise RuntimeError("Vocoder模型未能正确加载")
            
        # 确保vocoder在正确的设备上
        vocoder_model = ensure_model_on_device(vocoder_model, device, is_half)
        
        with torch.inference_mode():
            # 确保输入数据在正确的设备和精度上
            if is_half:
                cfm_res = cfm_res.half()
            cfm_res = cfm_res.to(device)
            
            # 生成波形
            wav_gen = vocoder_model(cfm_res)
            if wav_gen is None:
                raise RuntimeError("Vocoder生成的音频为None")
                
            audio = wav_gen[0][0]
            
            # 验证生成的音频
            if audio is None or not isinstance(audio, torch.Tensor):
                raise RuntimeError("生成的音频无效")
    except Exception as e:
        logger.error(f"Vocoder处理失败: {str(e)}")
        logger.exception(e)
        raise RuntimeError("音频生成失败") from e
    
    # 规范化音频
    max_audio = torch.abs(audio).max()
    if max_audio > 1:
        audio = audio / max_audio
    
    # 拼接静音
    audio_opt = torch.cat([torch.from_numpy(one_sec_silence).to(device), audio], 0)
    
    # 确定输出采样率
    if model_version in {"v1", "v2"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000  # v4
    
    # 转换为numpy数组
    audio_numpy = audio_opt.cpu().detach().numpy()
    
    # 确保是float32类型
    if audio_numpy.dtype == np.float16:
        audio_numpy = audio_numpy.astype(np.float32)
    
    # 缓存结果
    AUDIO_CACHE[cache_key] = (opt_sr, audio_numpy)
    
    return opt_sr, audio_numpy

# 验证API密钥
def verify_api_key(request: Request):
    """验证API密钥"""
    if not config.api_keys or config.api_keys == [""]:
        return True  # 未设置API密钥，不进行验证
    
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未提供有效的API密钥")
    
    api_key = auth_header.replace("Bearer ", "")
    if api_key not in config.api_keys:
        raise HTTPException(status_code=401, detail="API密钥无效")
    
    return True

# 获取参考音频的逻辑
def get_reference_audio_from_list(model_name: str, lang: str = "ZH") -> Tuple[str, str]:
    """
    从ASR输出目录中的列表文件中随机选择一个参考音频
    
    Args:
        model_name: 模型名称
        lang: 语言代码，默认为中文
        
    Returns:
        Tuple[str, str]: (参考音频路径, 参考文本)
    """
    asr_opt_dir = os.path.join(current_dir, "output", "asr_opt")
    list_file = os.path.join(asr_opt_dir, f"{model_name}.list")
    
    # 检查列表文件是否存在
    if not os.path.exists(list_file):
        logger.warning(f"模型 {model_name} 的列表文件不存在: {list_file}")
        return None, None
    
    # 读取列表文件
    references = []
    try:
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split("|")
                if len(parts) < 4:
                    continue
                
                audio_path, ref_model, ref_lang, ref_text = parts[0], parts[1], parts[2], parts[3]
                
                # 检查语言是否匹配
                # 如果请求的语言是auto，或者语言代码匹配（不区分大小写）
                if lang.upper() == "AUTO" or ref_lang.upper() == lang.upper():
                    # 将Windows路径转换为当前环境适用的路径
                    audio_path = audio_path.replace("\\", os.path.sep).replace("/", os.path.sep)
                    # 分离出最后两个路径部分（模型名/文件名）
                    path_parts = audio_path.split(os.path.sep)
                    if len(path_parts) >= 2:
                        relative_path = os.path.join("output", path_parts[-2], path_parts[-1])
                        if os.path.exists(relative_path):
                            references.append((relative_path, ref_text))
                        else:
                            # 如果文件不存在，尝试在当前目录下查找
                            alternative_path = os.path.join(current_dir, "output", path_parts[-2], path_parts[-1])
                            if os.path.exists(alternative_path):
                                references.append((alternative_path, ref_text))
    except Exception as e:
        logger.error(f"读取模型 {model_name} 的列表文件失败: {str(e)}")
        return None, None
    
    # 如果没有合适的参考音频，返回None
    if not references:
        logger.warning(f"模型 {model_name} 没有合适的参考音频")
        return None, None
    
    # 随机选择一个参考音频
    reference = random.choice(references)
    logger.info(f"为模型 {model_name} 随机选择参考音频: {reference[0]}")
    return reference

# 验证参考音频时长是否在3-10秒范围内
def validate_reference_audio(audio_path: str) -> bool:
    """
    验证参考音频时长是否在3-10秒范围内
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        bool: 是否有效
    """
    try:
        wav16k, sr = librosa.load(audio_path, sr=16000)
        duration = len(wav16k) / sr
        return 3 <= duration <= 10
    except Exception as e:
        logger.error(f"验证参考音频失败: {str(e)}")
        return False

# 修改初始化模型参考音频目录函数
def init_model_voice_dirs():
    """初始化每个模型的参考音频目录"""
    voices_base_dir = "voices"
    os.makedirs(voices_base_dir, exist_ok=True)
    
    # 获取所有模型
    voices = model_manager.get_all_voices()
    
    # 获取ASR输出目录中的列表文件
    asr_opt_dir = os.path.join(current_dir, "output", "asr_opt")
    if os.path.exists(asr_opt_dir):
        list_files = [f for f in os.listdir(asr_opt_dir) if f.endswith('.list')]
        logger.info(f"发现 {len(list_files)} 个ASR列表文件")
        
        # 将列表文件名（不含扩展名）添加到声音模型列表中
        for list_file in list_files:
            model_name = os.path.splitext(list_file)[0]
            if model_name not in [voice["name"] for voice in voices]:
                voices.append({"name": model_name})
    
    MAX_REFS_PER_MODEL = 3  # 每个模型最多保留的参考音频数量
    
    for voice in voices:
        voice_name = voice["name"] if isinstance(voice, dict) else voice
        voice_dir = os.path.join(voices_base_dir, voice_name)
        
        # 创建模型对应的参考音频目录
        if not os.path.exists(voice_dir):
            os.makedirs(voice_dir, exist_ok=True)
            logger.info(f"创建模型 {voice_name} 的参考音频目录: {voice_dir}")
        
        # 检查现有的参考音频
        existing_refs = []
        for file in os.listdir(voice_dir):
            if file.startswith('ref_') and file.endswith('.wav'):
                ref_path = os.path.join(voice_dir, file)
                # 验证现有参考音频是否有效
                if validate_reference_audio(ref_path):
                    existing_refs.append(file)
                else:
                    # 删除无效的参考音频及其文本文件
                    os.remove(ref_path)
                    text_file = ref_path + '.txt'
                    if os.path.exists(text_file):
                        os.remove(text_file)
        
        # 如果已有足够的有效参考音频，跳过添加新的
        if len(existing_refs) >= MAX_REFS_PER_MODEL:
            continue
        
        # 检查是否有列表文件，如果有则尝试添加参考音频
        ref_path, ref_text = get_reference_audio_from_list(voice_name)
        if ref_path and os.path.exists(ref_path) and validate_reference_audio(ref_path):
            # 生成唯一的文件名
            base_name = os.path.basename(ref_path)
            target_name = f"ref_{base_name}"
            target_path = os.path.join(voice_dir, target_name)
            
            # 检查是否已存在相同的参考音频
            if target_name not in existing_refs:
                try:
                    # 读取音频并保存
                    wav, sr = librosa.load(ref_path, sr=16000)
                    sf.write(target_path, wav, sr)
                    logger.info(f"为模型 {voice_name} 添加参考音频: {target_path}")
                    
                    # 创建一个文本文件保存参考文本
                    with open(os.path.join(voice_dir, f"{target_name}.txt"), "w", encoding="utf-8") as f:
                        f.write(ref_text)
                        
                    # 如果超过最大数量限制，删除最旧的参考音频
                    existing_refs.append(target_name)
                    if len(existing_refs) > MAX_REFS_PER_MODEL:
                        oldest_ref = sorted(existing_refs)[0]
                        oldest_path = os.path.join(voice_dir, oldest_ref)
                        if os.path.exists(oldest_path):
                            os.remove(oldest_path)
                            text_file = oldest_path + '.txt'
                            if os.path.exists(text_file):
                                os.remove(text_file)
                            logger.info(f"删除旧的参考音频: {oldest_path}")
                except Exception as e:
                    logger.error(f"处理参考音频失败: {str(e)}")
    
    # 确保有默认参考音频
    default_ref = os.path.join(voices_base_dir, "default_reference.wav")
    if not os.path.exists(default_ref):
        logger.info("创建默认参考音频")
        # 创建1秒16khz的默认音频
        sf.write(default_ref, np.random.rand(16000), 16000)

# 修改API启动函数，确保基础模型只加载一次
@app.on_event("startup")
async def startup_event():
    # 初始化基础模型
    init_models()
    
    # 初始化模型参考音频目录
    init_model_voice_dirs()
    
    # 直接加载默认模型
    voices = model_manager.get_all_voices()
    if voices:
        default_model = voices[0]["name"]
        logger.info(f"预加载默认模型: {default_model}")
        model_loader.load_model(default_model)

# 修改语音合成API
@app.post("/v1/audio/speech")
async def create_speech(request: Request, tts_request: TTSRequest, background_tasks: BackgroundTasks):
    # 验证API密钥
    verify_api_key(request)
    
    logger.info(f"接收TTS请求: {tts_request.model}, 文本长度: {len(tts_request.input)}")
    
    try:
        # 获取声音模型
        voice_model = None
        ref_wav_path = tts_request.voice
        prompt_text = None  # 参考文本
        
        # 检查是否是模型名称
        if tts_request.voice in model_manager.model_mapping:
            voice_model = tts_request.voice
            
            # 首先尝试从ASR列表文件中获取参考音频
            ref_path, ref_text = get_reference_audio_from_list(voice_model, tts_request.voice_language)
            
            if ref_path and os.path.exists(ref_path) and validate_reference_audio(ref_path):
                ref_wav_path = ref_path
                prompt_text = ref_text
                logger.info(f"使用ASR列表中的参考音频: {ref_wav_path}, 文本: {prompt_text}")
            else:
                # 尝试从模型的参考音频目录中获取
                voice_dir = os.path.join("voices", voice_model)
                if os.path.exists(voice_dir) and os.path.isdir(voice_dir):
                    # 查找目录中的wav文件作为参考
                    wav_files = [f for f in os.listdir(voice_dir) if f.endswith('.wav')]
                    if wav_files:
                        # 随机选择一个参考音频
                        ref_file = random.choice(wav_files)
                        ref_wav_path = os.path.join(voice_dir, ref_file)
                        
                        # 查找对应的文本文件
                        text_file = os.path.join(voice_dir, f"{ref_file}.txt")
                        if os.path.exists(text_file):
                            with open(text_file, "r", encoding="utf-8") as f:
                                prompt_text = f.read().strip()
                        
                        logger.info(f"使用模型目录中的参考音频: {ref_wav_path}")
            
            # 加载指定的模型
            if voice_model and voice_model in model_manager.model_mapping:
                model_loader.load_model(voice_model)
                
                # 检查模型版本是否支持推理
                if model_version in ["v1", "v2"]:
                    raise ValueError(f"不支持的模型版本: {model_version}。目前只支持v3和v4版本的模型进行推理。")
            
            # 如果仍然没有找到参考音频，使用默认参考音频
            if not os.path.exists(ref_wav_path) or not validate_reference_audio(ref_wav_path):
                ref_wav_path = "voices/default_reference.wav"
                if not os.path.exists(ref_wav_path):
                    # 创建默认参考音频
                    os.makedirs(os.path.dirname(ref_wav_path), exist_ok=True)
                    sf.write(ref_wav_path, np.random.rand(80000), 16000)
                    logger.info("创建默认参考音频")
                
        elif not os.path.exists(ref_wav_path):
            raise HTTPException(status_code=400, detail=f"找不到参考音频或声音模型: {ref_wav_path}")
        
        # 如果没有提供参考文本，使用默认文本或用户提供的文本
        if not prompt_text:
            prompt_text = tts_request.voice_text or "这是一段参考音频。"
        
        # 处理长文本，分段合成
        text_segments = split_text_for_tts(tts_request.input, tts_request.cut_method)
        if not text_segments:
            raise ValueError("文本分段后为空，无法合成")
            
        logger.info(f"文本分段: {len(text_segments)}段")
        
        # 分段合成音频
        all_audio_data = []
        sr = None
        
        for i, segment in enumerate(text_segments):
            logger.info(f"合成第{i+1}/{len(text_segments)}段文本: {segment[:30]}...")
            
            # 确保segment不为空
            if not segment or not segment.strip():
                logger.warning(f"跳过空文本段 {i+1}")
                continue
                
            # 合成语音
            segment_sr, segment_audio = synthesize_speech(
                ref_wav_path=ref_wav_path,
                prompt_text=prompt_text,
                prompt_language=tts_request.voice_language,
                text=segment,
                text_language=tts_request.text_language,
                top_k=tts_request.top_k,
                top_p=tts_request.top_p,
                temperature=tts_request.temperature,
                speed=tts_request.speed,
                sample_steps=tts_request.sample_steps,
                voice_model=voice_model,
            )
            
            if sr is None:
                sr = segment_sr
            elif sr != segment_sr:
                # 如果采样率不一致，进行重采样
                segment_audio = librosa.resample(segment_audio, orig_sr=segment_sr, target_sr=sr)
            
            all_audio_data.append(segment_audio)
        
        if not all_audio_data:
            raise ValueError("没有成功合成任何音频段")
            
        # 合并所有音频段
        if len(all_audio_data) > 1:
            # 添加0.3秒静音间隔
            silence = np.zeros(int(sr * 0.3))
            merged_audio = np.concatenate([np.concatenate([segment, silence]) for segment in all_audio_data[:-1]] + [all_audio_data[-1]])
        else:
            merged_audio = all_audio_data[0]
        
        # 处理音频格式
        audio_bytes = process_audio(merged_audio, sr, tts_request.response_format)
        
        # OpenAI API风格的响应
        if request.headers.get("accept") == "application/json":
            # 返回JSON响应，包含Base64编码的音频
            response = TTSResponse(
                model=tts_request.model,
                created=int(time.time()),
                audio=base64.b64encode(audio_bytes).decode("utf-8")
            )
            return response
        else:
            # 直接返回音频流
            content_type_map = {
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "ogg": "audio/ogg"
            }
            content_type = content_type_map.get(tts_request.response_format, "application/octet-stream")
            
            return StreamingResponse(
                iter([audio_bytes]),
                media_type=content_type,
                headers={"Content-Disposition": f"attachment; filename=speech.{tts_request.response_format}"}
            )
    
    except Exception as e:
        logger.error(f"TTS合成失败: {str(e)}")
        logger.exception(e)  # 打印完整异常堆栈
        raise HTTPException(status_code=400, detail=str(e))

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

# 模型信息端点
@app.get("/v1/models")
async def list_models(request: Request):
    verify_api_key(request)
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": int(time.time()) - 86400,  # 假设模型是1天前创建的
                "owned_by": "gpt-sovits"
            }
        ]
    }

# 添加一个API端点获取所有可用声音模型
@app.get("/v1/voices")
async def get_voices(request: Request):
    verify_api_key(request)
    voices = model_manager.get_all_voices()
    
    # 转换为OpenAI API格式
    result = {
        "object": "list",
        "data": [
            {
                "id": voice["name"],
                "name": voice["name"],
                "object": "voice",
                "created": int(time.time()) - 86400,  # 假设是1天前创建的
                "owned_by": "gpt-sovits",
                "iteration": voice["iteration"],
                "batch": voice["batch"]
            }
            for voice in voices
        ]
    }
    
    return result

# 添加加载模型的函数
def load_gpt_model(gpt_path):
    """加载GPT模型"""
    global t2s_model, max_sec, hz
    
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    gpt_config = dict_s1["config"]
    max_sec = gpt_config["data"]["max_sec"]
    
    # 卸载旧模型
    if t2s_model:
        t2s_model = t2s_model.cpu()
        try:
            torch.cuda.empty_cache()
        except:
            pass
    
    t2s_model = Text2SemanticLightningModule(gpt_config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    
    logger.info(f"GPT模型加载完成: {gpt_path}")

def load_sovits_model(sovits_path):
    """加载SoVITS模型"""
    global vq_model, hps, bigvgan_model, hifigan_model, model_version, version, if_lora_v3
    
    # 获取版本信息
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    logger.info(f"SoVITS模型版本: version={version}, model_version={model_version}, if_lora_v3={if_lora_v3}")
    
    # 检查是否为不支持的版本
    if model_version in ["v1", "v2"]:
        logger.warning(f"警告: 模型版本 {model_version} 不支持推理。目前只支持v3和v4版本的模型。")
    
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
    
    # 卸载旧模型
    if vq_model:
        vq_model = vq_model.cpu()
        try:
            torch.cuda.empty_cache()
        except:
            pass
    
    if model_version not in v3v4set:
        logger.warning(f"加载不支持推理的模型版本: {model_version}")
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
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
    
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    
    if if_lora_v3 == False:
        logger.info(f"加载SoVITS_{model_version}模型")
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
    else:
        path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
        path_sovits_v4 = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        
        logger.info(f"加载SoVITS_{model_version}预训练模型")
        vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False)
        
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        logger.info(f"加载SoVITS_{model_version}_lora{lora_rank}")
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        vq_model.eval()
    
    # 如果是v3模型，加载BigVGAN
    if model_version == "v3":
        init_bigvgan()
    # 如果是v4模型，加载HiFiGAN
    elif model_version == "v4":
        init_hifigan()
    
    logger.info(f"SoVITS模型加载完成: {sovits_path}")

# 添加上传参考音频的API端点
@app.post("/v1/voices/{voice_id}/reference")
async def upload_reference_audio(
    request: Request, 
    voice_id: str, 
    file: UploadFile = File(...),
    description: str = Form(None)
):
    """上传参考音频文件到指定的声音模型目录"""
    verify_api_key(request)
    
    # 检查模型是否存在
    if voice_id not in model_manager.model_mapping and not os.path.isdir(os.path.join("voices", voice_id)):
        raise HTTPException(status_code=404, detail=f"声音模型 {voice_id} 不存在")
    
    # 创建保存目录
    voice_dir = os.path.join("voices", voice_id)
    os.makedirs(voice_dir, exist_ok=True)
    
    # 生成唯一文件名
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension.lower() not in ['.wav', '.mp3', '.ogg']:
        raise HTTPException(status_code=400, detail="仅支持WAV、MP3或OGG格式的音频文件")
    
    unique_filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    file_path = os.path.join(voice_dir, unique_filename)
    
    # 保存上传的文件
    try:
        # 读取上传的文件内容
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 如果不是WAV格式，转换为WAV
        if file_extension.lower() != '.wav':
            # 使用librosa加载并转换为wav
            y, sr = librosa.load(file_path, sr=16000)
            sf.write(file_path, y, sr)
        
        # 裁剪音频到合适的长度 (3-10秒)
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) > 10 * sr:  # 如果超过10秒
            y = y[:10 * sr]  # 截取前10秒
        elif len(y) < 3 * sr:  # 如果少于3秒
            # 填充到3秒
            y = np.pad(y, (0, 3 * sr - len(y)), 'constant')
        
        # 保存处理后的音频
        sf.write(file_path, y, sr)
        
        return {
            "id": voice_id,
            "reference": unique_filename,
            "path": file_path,
            "description": description
        }
    
    except Exception as e:
        logger.error(f"上传参考音频失败: {str(e)}")
        # 删除可能部分写入的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

# 添加获取参考音频列表的API端点
@app.get("/v1/voices/{voice_id}/references")
async def get_voice_references(request: Request, voice_id: str):
    """获取指定声音模型的所有参考音频"""
    verify_api_key(request)
    
    voice_dir = os.path.join("voices", voice_id)
    if not os.path.exists(voice_dir) or not os.path.isdir(voice_dir):
        raise HTTPException(status_code=404, detail=f"声音模型 {voice_id} 的参考音频目录不存在")
    
    references = []
    for filename in os.listdir(voice_dir):
        if filename.endswith(('.wav', '.mp3', '.ogg')):
            file_path = os.path.join(voice_dir, filename)
            file_stat = os.stat(file_path)
            
            # 获取音频时长
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = len(y) / sr
            except:
                duration = 0
            
            references.append({
                "id": filename,
                "path": file_path,
                "size": file_stat.st_size,
                "created": int(file_stat.st_ctime),
                "duration": duration,
                "format": os.path.splitext(filename)[1][1:]  # 去掉点号
            })
    
    return {
        "object": "list",
        "data": references
    }

# 优化文本分段处理函数，处理空文本的情况
def split_text_for_tts(text, how_to_cut="凑四句一切"):
    """
    将长文本分段处理，便于TTS合成
    
    Args:
        text: 需要合成的文本
        how_to_cut: 分段方式
        
    Returns:
        List[str]: 分段后的文本列表
    """
    # 处理空文本
    if not text or len(text.strip()) == 0:
        logger.warning("输入文本为空，无法处理")
        return []
    
    text = text.strip()
    
    # 移植inference_webui.py中的文本分段逻辑
    if how_to_cut == "凑四句一切":
        # 每四个句子一组
        sentences = re.split(r'([。？！.?!])', text)
        new_sentences = []
        for i in range(0, len(sentences), 2):
            if i+1 < len(sentences):
                new_sentences.append(sentences[i] + sentences[i+1])
            else:
                new_sentences.append(sentences[i])
        
        result = []
        for i in range(0, len(new_sentences), 4):
            segment = ''.join(new_sentences[i:i+4])
            if segment.strip():  # 确保段落不为空
                result.append(segment)
        return result
        
    elif how_to_cut == "凑50字一切":
        # 每50个字符一组
        result = []
        for i in range(0, len(text), 50):
            segment = text[i:i+50]
            if segment.strip():  # 确保段落不为空
                result.append(segment)
        return result
        
    elif how_to_cut == "按中文句号。切":
        # 按中文句号切分
        sentences = text.split("。")
        result = []
        for s in sentences:
            if s.strip():  # 确保段落不为空
                result.append(s + "。")
        return result
        
    elif how_to_cut == "按英文句号.切":
        # 按英文句号切分
        sentences = text.split(".")
        result = []
        for s in sentences:
            if s.strip():  # 确保段落不为空
                result.append(s + ".")
        return result
        
    elif how_to_cut == "按标点符号切":
        # 按各种标点符号切分
        pattern = r'([。？！.?!,，])'
        sentences = re.split(pattern, text)
        result = []
        for i in range(0, len(sentences), 2):
            if i+1 < len(sentences):
                segment = sentences[i] + sentences[i+1]
                if segment.strip():  # 确保段落不为空
                    result.append(segment)
            elif sentences[i].strip():  # 处理最后一个元素
                result.append(sentences[i])
        return result
    
    # 默认不切分，直接返回整段文本
    return [text] if text.strip() else []

if __name__ == "__main__":
    # 启动FastAPI应用
    print(f"\n正在启动GPT-SoVITS API服务，请稍候...\n")
    
    import socket
    def get_ip_address():
        try:
            # 获取主机名
            hostname = socket.gethostname()
            # 获取IP地址
            ip_address = socket.gethostbyname(hostname)
            return ip_address
        except Exception:
            return "127.0.0.1"
    
    # 启动服务器
    host = config.host
    port = config.port
    
    # 如果host为0.0.0.0，则获取本机IP
    display_host = get_ip_address() if host == "0.0.0.0" else host
    
    # 使用uvicorn启动服务
    import threading
    def print_startup_message():
        # 延迟2秒确保服务已启动
        time.sleep(2)
        print("\n" + "="*50)
        print(f"✅ GPT-SoVITS API 服务启动成功!")
        print(f"🔗 API地址: http://{display_host}:{port}")
        print(f"📚 API文档: http://{display_host}:{port}/docs")
        print(f"🔍 健康检查: http://{display_host}:{port}/health")
        print("="*50 + "\n")
    
    # 在另一个线程中打印启动消息
    threading.Thread(target=print_startup_message).start()
    
    # 启动服务
    uvicorn.run(app, host=config.host, port=config.port) 