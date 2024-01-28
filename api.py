"""
# 1.28.2024 在原api.py基础上做出的一些改动

## 简介

- 原接口不变，仿照silero-api-server格式添加了一些endpoint，可接入傻酒馆sillytavern。
    - 运行api.py直至显示http://127.0.0.1:9880
    - 在staging版本的sillytavern>Extensions>TTS>Select TTS Provider选择silero
    - 将http://127.0.0.1:9880填入Provider Endpoint后点击reload
    - Select TTS Provider上方显示TTS Provider Loaded则连接成功，之后照常设置即可。

- 支持运行中根据讲话人名称自动更换声音模型或参考音频。
    - 如果运行api.py时使用-vd提供了声音模型根目录，可以根据讲话人名称（子文件夹名称或"default"）自动更换模型和参考音频。例如：
    
          python api.py -vd "D:/Voices"

    - 原有方式导入的声音被命名为“default”，可以使用原有方式更改其参考音频。也可以通过POST至 /set_default_models 更改default声音的模型，例如使用powershell新窗口运行：

          Invoke-RestMethod -Uri "http://127.0.0.1:9880/set_default_models" -Method Post -ContentType "application/json" -Body (@{gpt_path="D:\Voices\ZB\ZB.ckpt"; sovits_path="D:\Voices\ZB\ZB.pth"} | ConvertTo-Json)

- 默认输出语言是中文。可以在运行api.py时使用-ol指定其他输出语言，或后续POST至 /language 进行更改。
    - 例如要将输出语言改为英文，可以在新的powershell窗口运行：
    
          Invoke-RestMethod -Uri "http://127.0.0.1:9880/language" -Method Post -Body '{"language": "en"}' -ContentType "application/json"

## 声音模型根目录格式

    Voices
    ├─XXX
    ├   ├───XXX.ckpt
    ├   ├───XXX.pth
    ├   ├───XXX.wav
    ├   └───XXX.txt
    ├─YYY
    ├   ├───YYY.wav
    ├   └───YYY.txt
    ├─...
    ├
    └─ZZZ
        ├───ZZZ.ckpt
        ├───ZZZ.pth
        ├───ZZZ.wav
        └───ZZZ.txt

- 没有提供GPT和SoVITS模型文件的声音(例如上图的YYY)将使用原有方式指定的default声音模型。
- 每个文件夹中的txt文件是参考音频文本，目前仅支持单语言，内容格式为{语言}|{参考音频文本}，例如：

      zh|这是一段参考文本

## 新增的执行参数

`-vd`- `声音模型根目录，子文件夹以讲话人名称命名`
`-ol` - `输出音频语言, "中文","英文","日文","zh","en","ja"`




# api.py usage

` python api.py -dr "123.wav" -dt "一二三。" -dl "zh" `

## 执行参数:

`-s` - `SoVITS模型路径, 可在 config.py 中指定`
`-g` - `GPT模型路径, 可在 config.py 中指定`

调用请求缺少参考音频时使用
`-dr` - `默认参考音频路径`
`-dt` - `默认参考音频文本`
`-dl` - `默认参考音频语种, "中文","英文","日文","zh","en","ja"`

`-d` - `推理设备, "cuda","cpu","mps"`
`-a` - `绑定地址, 默认"127.0.0.1"`
`-p` - `绑定端口, 默认9880, 可在 config.py 中指定`
`-fp` - `覆盖 config.py 使用全精度`
`-hp` - `覆盖 config.py 使用半精度`

`-hb` - `cnhubert路径`
`-b` - `bert路径`

## 调用:

### 推理

endpoint: `/`

使用执行参数指定的参考音频:
GET:
    `http://127.0.0.1:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`
POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

手动指定当次推理所使用的参考音频:
GET:
    `http://127.0.0.1:9880?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh&text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh",
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
import os
import signal
import sys
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

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

g_config = global_config.Config()

# AVAILABLE_COMPUTE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-vd", "--voices_dir", type=str, default="", help="声音模型根目录，子文件夹以讲话人名称命名")
parser.add_argument("-ol", "--output_language", type=str, default="zh", help="输出音频语言")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")

parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")

parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu / mps")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False

parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")

args = parser.parse_args()

voices_dir = args.voices_dir
current_language=args.output_language
current_gpt_path=args.gpt_path
current_sovits_path=args.sovits_path

sovits_path = args.sovits_path
gpt_path = args.gpt_path


class DefaultRefer:
    def __init__(self, path, text, language):
        self.path = args.default_refer_path
        self.text = args.default_refer_text
        self.language = args.default_refer_language

    def is_ready(self) -> bool:
        return is_full(self.path, self.text, self.language)


class Voice:
    def __init__(self, folder):
        try:
            self.refer_wav_path = os.path.join(voices_dir, folder, f"{folder}.wav")
            if not os.path.isfile(self.refer_wav_path):
                raise ValueError("找不到参考音频 {refer_wav_path}")
            refer_txt_path = os.path.join(voices_dir, folder, f"{folder}.txt")
            if not os.path.isfile(refer_txt_path):
                raise ValueError("找不到参考文本 {refer_txt_path}")
            with open(refer_txt_path, 'r', encoding='utf-8') as file:
                content = file.read()
            parts = content.split('|', 1)
            if len(parts) == 2:
                self.refer_lang, self.refer_text = parts
            else:
                raise ValueError("参考文本格式错误。请使用'|'标注文本语言。目前仅支持单语言文本。例如:\nzh|这是一段参考文本。")
            self.sovits_path =os.path.join(voices_dir, folder, f"{folder}.pth")
            if not os.path.isfile(self.sovits_path):
                self.sovits_path=None
                print(f"[WARNING] 找不到 {folder} 专属SoVITS模型。此声音将使用默认SoVITS模型。")
            self.gpt_path = os.path.join(voices_dir, folder, f"{folder}.ckpt")
            if not os.path.isfile(self.gpt_path):
                self.gpt_path=None
                print(f"[WARNING] 找不到 {folder} 专属GPT模型。此声音将使用默认GPT模型。")
            self.name=folder
        except Exception as e:
            raise e


voices = {} 
if voices_dir!="":
    print(f"[INFO]  声音模型根目录: {voices_dir}")
    for folder in os.listdir(voices_dir):
        if os.path.isdir(os.path.join(voices_dir, folder)):
            try:
                voices[folder]=Voice(folder)
                print(f"[INFO] 根目录下发现声音: {folder}")
            except Exception as e:
                print(f"[WARNING]  {folder} 声音模型文件夹格式错误: {e}")
                pass

default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)

device = args.device
port = args.port
host = args.bind_addr

if sovits_path == "":
    sovits_path = g_config.pretrained_sovits_path
    print(f"[WARN] 未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
    gpt_path = g_config.pretrained_gpt_path
    print(f"[WARN] 未指定GPT模型路径, fallback后当前值: {gpt_path}")

# 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
if default_refer.path == "" or default_refer.text == "" or default_refer.language == "":
    default_refer.path, default_refer.text, default_refer.language = "", "", ""
    print("[INFO] 未指定默认参考音频")
else:
    print(f"[INFO] 默认参考音频路径: {default_refer.path}")
    print(f"[INFO] 默认参考音频文本: {default_refer.text}")
    print(f"[INFO] 默认参考音频语种: {default_refer.language}")

is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half  # 炒饭fallback

print(f"[INFO] 半精: {is_half}")

cnhubert_base_path = args.hubert_path
bert_path = args.bert_path

cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


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


n_semantic = 1024
dict_s2 = torch.load(sovits_path, map_location="cpu")
hps = dict_s2["config"]


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


hps = DictToAttrRecursive(hps)
hps.model.semantic_frame_rate = "25hz"
dict_s1 = torch.load(gpt_path, map_location="cpu")
config = dict_s1["config"]
ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
if is_half:
    vq_model = vq_model.half().to(device)
else:
    vq_model = vq_model.to(device)
vq_model.eval()
print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
hz = 50
max_sec = config['data']['max_sec']
t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
t2s_model.load_state_dict(dict_s1["weight"])
if is_half:
    t2s_model = t2s_model.half()
t2s_model = t2s_model.to(device)
t2s_model.eval()
total = sum([param.nelement() for param in t2s_model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec


dict_language = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja"
}


def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language):
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half == True else np.float32)
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if (is_half == True):
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
    phones1 = cleaned_text_to_sequence(phones1)
    texts = text.split("\n")
    audio_opt = []

    for text in texts:
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phones2 = cleaned_text_to_sequence(phones2)
        if (prompt_language == "zh"):
            bert1 = get_bert_feature(norm_text1, word2ph1).to(device)
        else:
            bert1 = torch.zeros((1024, len(phones1)), dtype=torch.float16 if is_half == True else torch.float32).to(
                device)
        if (text_language == "zh"):
            bert2 = get_bert_feature(norm_text2, word2ph2).to(device)
        else:
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
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
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if (is_half == True):
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = \
            vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0),
                            refer).detach().cpu().numpy()[
                0, 0]  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)


def handle_control(command):
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def handle_change(path, text, language):
    if is_empty(path, text, language):
        return JSONResponse({"code": 400, "message": '缺少任意一项以下参数: "path", "text", "language"'}, status_code=400)

    if path != "" or path is not None:
        default_refer.path = path
    if text != "" or text is not None:
        default_refer.text = text
    if language != "" or language is not None:
        default_refer.language = language

    print(f"[INFO] 当前默认参考音频路径: {default_refer.path}")
    print(f"[INFO] 当前默认参考音频文本: {default_refer.text}")
    print(f"[INFO] 当前默认参考音频语种: {default_refer.language}")
    print(f"[INFO] is_ready: {default_refer.is_ready()}")

    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


def handle_load(new_gpt_path, new_sovits_path):
    global gpt_path,sovits_path,current_gpt_path,current_sovits_path
    if(new_gpt_path=="" or new_gpt_path is None):
        new_gpt_path = gpt_path
        if(gpt_path=="" or gpt_path is None):
            print("[ERROR] 未设置默认GPT模型地址")
            raise ValueError("未设置默认GPT模型地址")
    if(new_sovits_path=="" or new_sovits_path is None):
        new_sovits_path = sovits_path
        if(sovits_path=="" or sovits_path is None):
            print("[ERROR] 未设置默认SoVITS模型地址")
            raise ValueError("未设置默认SoVITS模型地址")
    if(os.path.normpath(os.path.abspath(current_gpt_path))==os.path.normpath(os.path.abspath(new_gpt_path))
       and os.path.normpath(os.path.abspath(current_sovits_path)) == os.path.normpath(os.path.abspath(new_sovits_path))):
        return
    print(f"current models: {current_gpt_path}, {current_sovits_path}")
    print(f"loading new models: {new_gpt_path}, {new_sovits_path}")
    current_gpt_path=new_gpt_path
    current_sovits_path=new_sovits_path
    
    
    global dict_s2, hps, dict_s1, config, vq_model, max_sec, t2s_model
    dict_s2 = torch.load(new_sovits_path, map_location="cpu")  # Corrected the variable name here
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    dict_s1 = torch.load(new_gpt_path, map_location="cpu")
    config = dict_s1["config"]
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    max_sec = config['data']['max_sec']
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()


def handle(refer_wav_path, prompt_text, prompt_language, text, text_language):
    if (
            refer_wav_path == "" or refer_wav_path is None
            or prompt_text == "" or prompt_text is None
            or prompt_language == "" or prompt_language is None
    ):
        refer_wav_path, prompt_text, prompt_language = (
            default_refer.path,
            default_refer.text,
            default_refer.language,
        )
        if not default_refer.is_ready():
            return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)

    with torch.no_grad():
        gen = get_tts_wav(
            refer_wav_path, prompt_text, prompt_language, text, text_language
        )
        sampling_rate, audio_data = next(gen)

    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)

    torch.cuda.empty_cache()
    if(device=="mps"): #added condition so it can run on my device for testing
        torch.mps.empty_cache()
    return StreamingResponse(wav, media_type="audio/wav")


app = FastAPI()


@app.post("/control")
async def control(request: Request):
    json_post_raw = await request.json()
    return handle_control(json_post_raw.get("command"))


@app.get("/control")
async def control(command: str = None):
    return handle_control(command)


@app.post("/set_default_models")
async def set_default_models(request: Request):
    global gpt_path,sovits_path
    json_post_raw = await request.json()
    gpt_path = json_post_raw.get("gpt_path")
    sovits_path = json_post_raw.get("sovits_path")
    return JSONResponse({"gpt_path":gpt_path,"sovits_path":sovits_path},status_code=200)


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
    )


@app.get("/")
async def tts_endpoint(
        refer_wav_path: str = None,
        prompt_text: str = None,
        prompt_language: str = None,
        text: str = None,
        text_language: str = None,
):
    return handle(refer_wav_path, prompt_text, prompt_language, text, text_language)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/speakers")
async def speakers(request:Request):
    voices_info = [
        {
            "name":"default",
            "voice_id":"default",
            "preview_url": f"{str(request.base_url)}sample/default"
        }
    ]
    if(len(voices)>0):
        for v in voices.values():
            voices_info.append(
                {
                    "name":v.name,
                    "voice_id":v.name,
                    "preview_url": f"{str(request.base_url)}sample/{v.name}"
                } 
            )
    return voices_info


@app.post("/generate")
async def generate(request:Request):
    json_post_raw=await request.json()
    speaker = json_post_raw.get("speaker")
    if(speaker=="default"):
        handle_load(gpt_path,sovits_path)
        handle_result=handle(
            None,
            None,
            None,
            json_post_raw.get("text"),
            current_language,
        )
    else:
        handle_load(voices[speaker].gpt_path,voices[speaker].sovits_path)
        return handle(
            voices[speaker].refer_wav_path,
            voices[speaker].refer_text,
            voices[speaker].refer_lang,
            json_post_raw.get("text"),
            current_language,
        )


@app.get("/sample/{speaker}")
async def play_sample(speaker: str = 'default'):
    if(speaker=='default'):
        return FileResponse(default_refer.path,status_code=200)
    print(f"sending {voices[speaker].refer_wav_path}")
    return FileResponse(voices[speaker].refer_wav_path,status_code=200)


@app.post("/session") #just a placeholder
async def session(request:Request):
    return JSONResponse({},status_code=200)

@app.get("/language")
async def get_languages():
    return JSONResponse(list(dict_language.keys()),headers={'Content-Type': 'text/plain; charset=utf-8'}, status_code=200)

@app.post("/language")
async def set_language(request: Request):
    global current_language
    json_post_raw=await request.json()
    current_language = json_post_raw.get("language")
    print(f"[INFO] output language is set to:{current_language}")
    return JSONResponse(f"current language: {current_language}",status_code=200)



if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, workers=1)
