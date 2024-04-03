"""
# api.py usage

` python api.py -dr "123.wav" -dt "一二三。" -dl "zh" `

## 执行参数:

`-s` - `SoVITS模型路径, 可在 config.py 中指定`
`-g` - `GPT模型路径, 可在 config.py 中指定`

调用请求缺少参考音频时使用
`-dr` - `默认参考音频路径`
`-dt` - `默认参考音频文本`
`-dl` - `默认参考音频语种, "中文","英文","日文","zh","en","ja"`

`-d` - `推理设备, "cuda","cpu"`
`-a` - `绑定地址, 默认"127.0.0.1"`
`-p` - `绑定端口, 默认9880, 可在 config.py 中指定`
`-fp` - `覆盖 config.py 使用全精度`
`-hp` - `覆盖 config.py 使用半精度`
`-sm` - `流式返回模式, 默认不启用, "close","c", "normal","n", "keepalive","k"`
`-mt` - `返回的音频编码格式, 流式默认ogg, 非流式默认wav, "wav", "ogg", "aac"`
`-cp` - `文本切分符号设定, 默认为空, 以",.，。"字符串的方式传入`
`-bs` - `批处理大小,默认为1`
`-rf` - `碎片返回,约等于流`
`-sb` - `分桶处理,可能可以减少计算量,与碎片返回冲突`

`-hb` - `cnhubert路径`
`-b` - `bert路径`

## 调用:

### 推理

endpoint: `/`

使用执行参数指定的参考音频:
GET:
    `http://127.0.0.1:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh` #从zh,en,ja,auto中选择
POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh" #从zh,en,ja,auto中选择
}
```

使用执行参数指定的参考音频并设定分割符号:
GET:
    `http://127.0.0.1:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh&cut_punc=，。`
POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh", #从zh,en,ja,auto中选择
    "cut_punc": "，。",
}
```

手动指定当次推理所使用的参考音频:
GET:
    `http://127.0.0.1:9880?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh&text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh` #从zh,en,ja,auto中选择
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh", #从zh,en,ja,auto中选择
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400


### 更换默认参考音频

endpoint: `/change_refer`

key与推理端一样

GET:
    `http://127.0.0.1:9880/change_refer?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh"
}
```

RESP:
成功: json, http code 200
失败: json, 400


### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
    `http://127.0.0.1:9880/control?command=restart`
POST:
```json
{
    "command": "restart"
}
```

RESP: 无

"""

import argparse
import os,re
import sys


now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir)) # 神奇位置,防止import的问题


import signal
import LangSegment
from time import time as ttime
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from feature_extractor import cnhubert
from io import BytesIO
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
import config as global_config
import logging
import subprocess
from typing import Dict, List, Tuple
from tools.i18n.i18n import I18nAuto
import traceback
import math
i18n = I18nAuto()



def is_empty(*items):  # 任意一项不为空返回False
    for item in items:
        if item is not None and item != "":
            return False
    return True


def is_full(*items):  # 任意一项为空返回False
    for item in items:
        if item is None or item == "":
            return False
    return True


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config, is_fast_inference
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    try:
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False, flash_attn_enabled=flash_atten)
        is_fast_inference = True
    except TypeError:
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        is_fast_inference = False
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    logger.info("Number of parameter: %.2fM" % (total / 1e6))


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=precision,
        ).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=precision,
        ).to(device)

    return bert


def get_phones_and_bert(text:str,language:str):
    if language in {"en","all_zh","all_ja"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=precision,
            ).to(device)
    elif language in {"zh", "ja","auto"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        # logger.info(textlist)
        # logger.info(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones,bert.to(torch.float16 if is_half == True else torch.float32),norm_text


def extract_feature_for_text(textlist:list, langlist:list)->Tuple[list, torch.Tensor, str]:
    if len(textlist) == 0:
        return None, None, None
        
    phones, bert_features, norm_text = get_phones_and_bert(textlist, langlist)
    return phones, bert_features, norm_text


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


class REF:
    def __init__(self, ref_path="", ref_text="", ref_language=""):
        ref_text = ref_text.strip("\n")
        if ref_text:
            if (ref_text[-1] not in splits): ref_text += "。" if ref_language != "en" else "."
        if ref_language:
            ref_language = dict_language[ref_language.lower()]
        self.path = ref_path
        self.text = ref_text
        self.language = ref_language

    def set_prompt_semantic(self, ref_wav_path:str):
        zero_wav = np.zeros(
            int(hps.data.sampling_rate * 0.3),
            dtype=np.float16 if is_half else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
            if is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()

            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = vq_model.extract_latent(ssl_content)
    
            prompt_semantic = codes[0, 0].to(device)
            self.prompt_semantic = prompt_semantic
            self.codes = codes
            self.ssl_content = ssl_content

    def set_ref_spec(self, ref_audio_path):
        audio = load_audio(ref_audio_path, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
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
        spec = spec.to(device)
        if is_half:
            spec = spec.half()
        # self.refer_spec = spec
        self.refer_spec = spec
    
    def set_ref_audio(self):
        '''
            To set the reference audio for the TTS model, 
                including the prompt_semantic and refer_spec.
            Args:
                ref_audio_path: str, the path of the reference audio.
        '''
        self.set_prompt_semantic(self.path)
        self.set_ref_spec(self.path)
        self.phone, self.bert_feature, self.norm_text = get_phones_and_bert(self.text, self.language)
    
    def is_ready(self) -> bool:
        return is_full(self.path, self.text, self.language)


def pack_audio(audio_bytes, data, rate):
    if media_type == "ogg":
        audio_bytes = pack_ogg(audio_bytes, data, rate)
    elif media_type == "aac":
        audio_bytes = pack_aac(audio_bytes, data, rate)
    else:
        # wav无法流式, 先暂存raw
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
    data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int16)
    wav_bytes = BytesIO()
    sf.write(wav_bytes, data, rate, format='wav')

    return wav_bytes


def pack_aac(audio_bytes, data, rate):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    audio_bytes.write(out)

    return audio_bytes


def read_clean_buffer(audio_bytes):
    audio_chunk = audio_bytes.getvalue()
    audio_bytes.truncate(0)
    audio_bytes.seek(0)

    return audio_bytes, audio_chunk


def cut_text(text, punc):
    punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        # 在句子不存在符号或句尾无符号的时候保证文本完整
        if len(items)%2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    return text


def only_punc(text):
    return not any(t.isalnum() or t.isalpha() for t in text)


def preprocess(text:list, lang:str)->List[Dict]:
    result = []
    for _text in text:
        phones, bert_features, norm_text = extract_feature_for_text(_text, lang)
        if phones is None:
            continue
        res={
            "phones": phones,
            "bert_features": bert_features,
            "norm_text": norm_text,
        }
        result.append(res)
    return result


def audio_postprocess( 
                      audio:List[torch.Tensor], 
                      sr:int, 
                      batch_index_list:list=None, 
                      fragment_interval:float=0.3
                      ):
    zero_wav = torch.zeros(
                    int(hps.data.sampling_rate * fragment_interval),
                    dtype=precision,
                    device=device
                )
        
    audio_bytes = BytesIO()

    for i, batch in enumerate(audio):
        for j, audio_fragment in enumerate(batch):
            max_audio=torch.abs(audio_fragment).max()#简单防止16bit爆音
            if max_audio>1: audio_fragment/=max_audio
            audio_fragment:torch.Tensor = torch.cat([audio_fragment, zero_wav], dim=0)
            audio[i][j] = audio_fragment.cpu().numpy()
            
        
    if split_bucket:
        audio = recovery_order(audio, batch_index_list)
    else:
        # audio = [item for batch in audio for item in batch]
        audio = sum(audio, [])
            
            
    audio = pack_audio(audio_bytes,(np.concatenate(audio, 0) * 32768).astype(np.int16),hps.data.sampling_rate)

    if media_type == "wav":
        audio_bytes = pack_wav(audio,hps.data.sampling_rate)
    return audio_bytes.getvalue() 


def batch_sequences(sequences: List[torch.Tensor], axis:int = 0, pad_value:int = 0, max_length:int=None):
        seq = sequences[0]
        ndim = seq.dim()
        if axis < 0:
            axis += ndim
        dtype:torch.dtype = seq.dtype
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


def to_batch(data:list, ref:REF, 
             threshold:float=0.75, 
             ):
        
        _data:list = []
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
            while pos <index_and_len_list.shape[0]:
                # batch_index_list.append(index_and_len_list[pos:min(pos+batch_size,len(index_and_len_list))])
                pos_end = min(pos+batch_size,index_and_len_list.shape[0])
                while pos < pos_end:
                    batch=index_and_len_list[pos:pos_end, 1].astype(np.float32)
                    score=batch[(pos_end-pos)//2]/(batch.mean()+1e-8)
                    if (score>=threshold) or (pos_end-pos==1):
                        batch_index=index_and_len_list[pos:pos_end, 0].tolist()
                        batch_index_list_len += len(batch_index)
                        batch_index_list.append(batch_index)
                        pos = pos_end
                        break
                    pos_end=pos_end-1
            
            assert batch_index_list_len == len(data)
            
        else:
            for i in range(len(data)):
                if i%batch_size == 0:
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
            bert_max_len = 0
            phones_max_len = 0
            for item in item_list:
                all_bert_features = torch.cat([ref.bert_feature, item["bert_features"]], 1).to(dtype=precision, device=device)
                all_phones = torch.LongTensor(ref.phone+item["phones"]).to(device)
                phones = torch.LongTensor(item["phones"]).to(device)
                # norm_text = ref.norm_text+item["norm_text"]

                bert_max_len = max(bert_max_len, all_bert_features.shape[-1])
                phones_max_len = max(phones_max_len, phones.shape[-1])
                
                phones_list.append(phones)
                phones_len_list.append(phones.shape[-1])
                all_phones_list.append(all_phones)
                all_phones_len_list.append(all_phones.shape[-1])
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])
                
            phones_batch = phones_list
            all_phones_batch = all_phones_list
            all_bert_features_batch = all_bert_features_list
            
            
            batch = {
                "phones": phones_batch,
                "phones_len": torch.LongTensor(phones_len_list).to(device),
                "all_phones": all_phones_batch,
                "all_phones_len": torch.LongTensor(all_phones_len_list).to(device),
                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch
            }
            _data.append(batch)
        
        return _data, batch_index_list


def recovery_order(data:list, batch_index_list:list)->list:
    '''
    Recovery the order of the audio according to the batch_index_list.
    
    Args:
        data (List[list(np.ndarray)]): the out of order audio .
        batch_index_list (List[list[int]]): the batch index list.
    
    Returns:
        list (List[np.ndarray]): the data in the original order.
    '''
    length = len(sum(batch_index_list, []))
    _data = [None]*length
    for i, index_list in enumerate(batch_index_list):
        for j, index in enumerate(index_list):
            _data[index] = data[i][j]
    return _data


def run(ref:REF, text, text_lang):
        logger.info("run")

        ########## variables initialization ###########
        top_k = 5
        top_p = 1
        temperature = 1
        batch_threshold = 0.75
        fragment_interval = 0.3
        text_lang = dict_language[text_lang.lower()]


        if ref.path in [None, ""] or \
            ((ref.prompt_semantic is None) or (ref.refer_spec is None)):
            raise ValueError("ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()")


        t0 = ttime()
        ###### text preprocessing ########
        t1 = ttime()
        data:list = None
        if not return_fragment:
            data = text.split("\n")
            if len(data) == 0:
                yield np.zeros(int(hps.data.sampling_rate), type=np.int16)
                return
            
            batch_index_list:list = None
            data = preprocess(data, text_lang)
            data, batch_index_list = to_batch(data, ref, 
                                threshold=batch_threshold,
            )
        else:
            texts = text.split("\n")
            data = []
            for i in range(len(texts)):
                if i%batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])
                
            def make_batch(batch_texts):
                batch_data = []
                batch_data = preprocess(batch_texts, text_lang)
                if len(batch_data) == 0:
                    return None
                batch, _ = to_batch(batch_data, ref,
                            threshold=batch_threshold,
                            )
                return batch[0]
            
        t2 = ttime()
        try:
            ###### inference ######
            t_34 = 0.0
            t_45 = 0.0
            audio = []
            for item in data:
                t3 = ttime()
                if return_fragment:
                    item = make_batch(item)
                    if item is None:
                        continue
                    
                batch_phones:List[torch.LongTensor] = item["phones"]
                batch_phones_len:torch.LongTensor = item["phones_len"]
                all_phoneme_ids:List[torch.LongTensor] = item["all_phones"]
                all_phoneme_lens:torch.LongTensor  = item["all_phones_len"]
                all_bert_features:List[torch.LongTensor] = item["all_bert_features"]
                norm_text:str = item["norm_text"]
        
                print(norm_text)
                
                prompt = ref.prompt_semantic.expand(len(all_phoneme_ids), -1).to(device)
                
                with torch.no_grad():
                    pred_semantic_list, idx_list = t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_lens,
                        prompt,
                        all_bert_features,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=hz * max_sec,
                    )
                t4 = ttime()
                t_34 += t4 - t3
                
                refer_audio_spec:torch.Tensor = ref.refer_spec.to(dtype=precision, device=device)
                    
                batch_audio_fragment = []

                pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                upsample_rate = math.prod(vq_model.upsample_rates)
                audio_frag_idx = [pred_semantic_list[i].shape[0]*2*upsample_rate for i in range(0, len(pred_semantic_list))]
                audio_frag_end_idx = [ sum(audio_frag_idx[:i+1]) for i in range(0, len(audio_frag_idx))]
                all_pred_semantic = torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(device)
                _batch_phones = torch.cat(batch_phones).unsqueeze(0).to(device)
                _batch_audio_fragment = (vq_model.decode(
                        all_pred_semantic, _batch_phones,refer_audio_spec
                    ).detach()[0, 0, :])
                audio_frag_end_idx.insert(0, 0)
                batch_audio_fragment= [_batch_audio_fragment[audio_frag_end_idx[i-1]:audio_frag_end_idx[i]] for i in range(1, len(audio_frag_end_idx))]
            
                
                t5 = ttime()
                t_45 += t5 - t4
                if return_fragment:
                    logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t4 - t3, t5 - t4))
                    yield audio_postprocess([batch_audio_fragment], 
                                                    hps.data.sampling_rate, 
                                                    None, 
                                                    fragment_interval
                                                    )
                else:
                    audio.append(batch_audio_fragment)
                       
            logger.info("return_fragment:"+str(return_fragment)+"  split_bucket:"+str(split_bucket)+"  batch_size"+str(batch_size)+"  media_type:"+media_type)
            if not return_fragment:
                logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t_34, t_45))
                yield audio_postprocess(audio, 
                                            hps.data.sampling_rate, 
                                            batch_index_list, 
                                            fragment_interval
                                                )

        except Exception as e:
            traceback.print_exc()
            # 必须返回一个空音频, 否则会导致显存不释放。
            yield np.zeros(int(hps.data.sampling_rate), dtype=np.int16)
        finally:
            pass


def get_tts_wav(ref:REF, text, text_language):
    logger.info("get_tts_wav")
    t0 = ttime()
    t1 = ttime()
    text_language = dict_language[text_language.lower()]
    phones1, bert1, norm_text1 = ref.phone, ref.bert_feature, ref.norm_text
    texts = text.split("\n")
    audio_bytes = BytesIO()

    for text in texts:
        # 简单防止纯符号引发参考音频泄露
        if only_punc(text):
            continue
        print(text)

        audio_opt = []
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = ref.prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()

        zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half == True else np.float32)

        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=config['inference']['top_k'],
                early_stop_num=hz * max_sec)
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        if isinstance(pred_semantic, list) and isinstance(pred_semantic, list): # 神秘代码,有些时候sys.path会出问题,import的是fast inference分支的AR
            pred_semantic = pred_semantic[0]
            idx=idx[0]
            pred_semantic = pred_semantic[-idx:]
            pred_semantic = pred_semantic.unsqueeze(0).unsqueeze(0)
        else:
            pred_semantic = pred_semantic[:,-idx:]
            pred_semantic = pred_semantic.unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次

        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = \
            vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0),
                            ref.refer_spec).detach().cpu().numpy()[0, 0]  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
        audio_bytes = pack_audio(audio_bytes,(np.concatenate(audio_opt, 0) * 32768).astype(np.int16),hps.data.sampling_rate)
        logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        if return_fragment:
            audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
            yield audio_chunk
    
    if not return_fragment: 
        if media_type == "wav":
            audio_bytes = pack_wav(audio_bytes,hps.data.sampling_rate)
        yield audio_bytes.getvalue()

# --------------------------------
# 初始化部分
# --------------------------------


# logger
logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logger = logging.getLogger('uvicorn')

# 获取配置
g_config = global_config.Config()

# 获取参数
parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")
parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")
parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-bs", "--batch_size", type=int, default=1, help="批处理大小")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")
parser.add_argument("-rf", "--return_fragment", action="store_true", default=False, help="是否开启碎片返回")
parser.add_argument("-sb", "--split_bucket", action="store_true", default=False, help="是否将批处理分成多个桶")
parser.add_argument("-fa", "--flash_atten", action="store_true", default=False, help="是否开启flash_attention")
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False
parser.add_argument("-mt", "--media_type", type=str, default="wav", help="音频编码格式, wav / ogg / aac")
parser.add_argument("-cp", "--cut_punc", type=str, default="", help="文本切分符号设定, 符号范围,.;?!、，。？！;：…")
# 切割常用分句符为 `python ./api.py -cp ".?!。？！"`
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
batch_size = args.batch_size
return_fragment = args.return_fragment
split_bucket = args.split_bucket
flash_atten = args.flash_atten

dict_language = {
    "中文": "all_zh",
    "英文": "en",
    "英语": "en",
    "日文": "all_ja",
    "日语": "all_ja",
    "中英混合": "zh",
    "日英混合": "ja",
    "多语种混合": "auto",    #多语种启动切分识别语种
    "all_zh": "all_zh",
    "en": "en",
    "all_ja": "all_ja",
    "zh": "zh",
    "ja": "ja",
    "auto": "auto",
}
splits = [",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"]
is_fast_inference = True


# 模型路径检查
if sovits_path == "":
    sovits_path = g_config.pretrained_sovits_path
    logger.warn(f"未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
    gpt_path = g_config.pretrained_gpt_path
    logger.warn(f"未指定GPT模型路径, fallback后当前值: {gpt_path}")

# 获取半精度
is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half  # 炒饭fallback
logger.info(f"半精: {is_half}")

precision = torch.float16 if is_half else torch.float32
device = torch.device(device)


# 音频编码格式
if args.media_type.lower() in ["aac","ogg"]:
    media_type = args.media_type.lower()
elif not return_fragment:
    media_type = "wav"
else:
    media_type = "ogg"
logger.info(f"编码格式: {media_type}")


# 初始化模型
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
change_sovits_weights(sovits_path)
change_gpt_weights(gpt_path)


# ?????
if return_fragment:
    split_bucket = False
    logger.info("碎片返回已开启")
    logger.info("分桶处理已关闭")

if split_bucket and is_fast_inference:
    logger.info("碎片返回已开启")

if batch_size != 1 and is_fast_inference:
    logger.info("批处理已开启")
    logger.info(f"批处理大小:{batch_size}")
else:
    logger.info("批处理已关闭")


# 应用参数配置
default_refer = REF(args.default_refer_path, args.default_refer_text, args.default_refer_language)


# 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
if not default_refer.is_ready():
    default_refer.path, default_refer.text, default_refer.language = "", "", ""
    logger.info("未指定默认参考音频")
else:
    logger.info(f"默认参考音频路径: {default_refer.path}")
    logger.info(f"默认参考音频文本: {default_refer.text}")
    logger.info(f"默认参考音频语种: {default_refer.language}")
    default_refer.set_ref_audio()


def handle_control(command):
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def handle_change(path, text, language):
    global default_refer
    if is_empty(path, text, language):
        return JSONResponse({"code": 400, "message": '缺少任意一项以下参数: "path", "text", "language"'}, status_code=400)

    if (path != "" or path is not None) and\
       (text != "" or text is not None) and\
       (language != "" or language is not None):
        default_refer = REF(path, text, language)

    logger.info(f"当前默认参考音频路径: {default_refer.path}")
    logger.info(f"当前默认参考音频文本: {default_refer.text}")
    logger.info(f"当前默认参考音频语种: {default_refer.language}")
    logger.info(f"is_ready: {default_refer.is_ready()}")


    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


def handle(refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc):    
    if (refer_wav_path != default_refer.path) or\
    (prompt_text != default_refer.text) or\
    (prompt_language != default_refer.language):
        ref = REF(refer_wav_path, prompt_text, prompt_language)
    else:
        ref = default_refer

    if (
            refer_wav_path == "" or refer_wav_path is None
            or prompt_text == "" or prompt_text is None
            or prompt_language == "" or prompt_language is None
    ):
        ref = default_refer
        if not default_refer.is_ready():
            return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)

    if cut_punc == None:
        text = cut_text(text,default_cut_punc)
    else:
        text = cut_text(text,cut_punc)
    

    
    if is_fast_inference:
        return StreamingResponse(run(ref, text,text_language), media_type="audio/"+media_type)
    else:
        return StreamingResponse(get_tts_wav(ref, text,text_language), media_type="audio/"+media_type)


# --------------------------------
# 接口部分
# --------------------------------
app = FastAPI()

@app.post("/set_model")
async def set_model(request: Request):
    json_post_raw = await request.json()
    global gpt_path
    gpt_path=json_post_raw.get("gpt_model_path")
    global sovits_path
    sovits_path=json_post_raw.get("sovits_model_path")
    logger.info("gptpath"+gpt_path+";vitspath"+sovits_path)
    change_sovits_weights(sovits_path)
    change_gpt_weights(gpt_path)
    return "ok"


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
async def change_refer(
        refer_wav_path: str = None,
        prompt_text: str = None,
        prompt_language: str = None
):
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
    )


@app.get("/")
async def tts_endpoint(
        refer_wav_path: str = "",
        prompt_text: str = "",
        prompt_language: str = "",
        text: str = "",
        text_language: str = "",
        cut_punc: str = "",
):
    return handle(refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc)


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, workers=1)
