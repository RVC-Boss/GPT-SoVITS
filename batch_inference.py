import argparse
import os
import pdb
import signal
import sys
from time import time as ttime
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
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

g_config = global_config.Config()

# AVAILABLE_COMPUTE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")

parser.add_argument("-dr", "--default_refer_path", type=str, default="",
                    help="默认参考音频路径, 请求缺少参考音频时调用")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")

parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False

parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")

args = parser.parse_args()

sovits_path = args.sovits_path
gpt_path = args.gpt_path

default_refer_path = args.default_refer_path
default_refer_text = args.default_refer_text
default_refer_language = args.default_refer_language
has_preset = False

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
if default_refer_path == "" or default_refer_text == "" or default_refer_language == "":
    default_refer_path, default_refer_text, default_refer_language = "", "", ""
    print("[INFO] 未指定默认参考音频")
    has_preset = False
else:
    print(f"[INFO] 默认参考音频路径: {default_refer_path}")
    print(f"[INFO] 默认参考音频文本: {default_refer_text}")
    print(f"[INFO] 默认参考音频语种: {default_refer_language}")
    has_preset = True

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
dict_s2 = torch.load(sovits_path, map_location="cpu", weights_only=False)
hps = dict_s2["config"]
print(hps)

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


hps = DictToAttrRecursive(hps)
hps.model.semantic_frame_rate = "25hz"
dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
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
t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
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
        wav16k=torch.cat([wav16k,zero_wav_torch])
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
    # yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    return hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
def get_tts_wavs(ref_wav_path, prompt_text, prompt_language, textss, text_language):
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
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
        wav16k=torch.cat([wav16k,zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
    phones1 = cleaned_text_to_sequence(phones1)
    audios_opt=[]
    for text0 in textss:
        texts = text0.strip("\n").split("\n")
        audio_opt = []
        for text in texts:
            text=text.strip("。")+"。"
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
        audios_opt.append([text0,(np.concatenate(audio_opt, 0) * 32768).astype(np.int16)])
    return audios_opt


# get_tts_wav(r"D:\BaiduNetdiskDownload\gsv\speech\萧逸声音-你得先从滑雪的基本技巧学起.wav", "你得先从滑雪的基本技巧学起。", "中文", "我觉得还是该给喜欢的女孩子一场认真的告白。", "中文")
# with open(r"D:\BaiduNetdiskDownload\gsv\烟嗓-todo1.txt","r",encoding="utf8")as f:
# with open(r"D:\BaiduNetdiskDownload\gsv\年下-todo1.txt","r",encoding="utf8")as f:
# with open(r"D:\BaiduNetdiskDownload\gsv\萧逸3b.txt","r",encoding="utf8")as f:
with open(r"D:\BaiduNetdiskDownload\gsv\萧逸4.txt","r",encoding="utf8")as f:
    textss=f.read().split("\n")
for idx,(text,audio)in enumerate(get_tts_wavs(r"D:\BaiduNetdiskDownload\gsv\speech\萧逸声音-你得先从滑雪的基本技巧学起.wav", "你得先从滑雪的基本技巧学起。", "中文", textss, "中文")):

# for idx,(text,audio)in enumerate(get_tts_wavs(r"D:\BaiduNetdiskDownload\gsv\足够的能力，去制定好自己的生活规划。低沉烟嗓.MP3_1940480_2095360.wav", "足够的能力，去制定好自己的生活规划。", "中文", textss, "中文")):
# for idx,(text,audio)in enumerate(get_tts_wavs(r"D:\BaiduNetdiskDownload\gsv\不会呀！你前几天才吃过你还说好吃来着。年下少年音.MP3_537600_711040.wav", "不会呀！你前几天才吃过你还说好吃来着。", "中文", textss, "中文")):
    print(idx,text)
    # sf.write(r"D:\BaiduNetdiskDownload\gsv\output\烟嗓第一批\%04d-%s.wav"%(idx,text),audio,32000)
    # sf.write(r"D:\BaiduNetdiskDownload\gsv\output\年下\%04d-%s.wav"%(idx,text),audio,32000)
    sf.write(r"D:\BaiduNetdiskDownload\gsv\output\萧逸第4批\%04d-%s.wav"%(idx,text),audio,32000)


# def handle(command, refer_wav_path, prompt_text, prompt_language, text, text_language):
#     if command == "/restart":
#         os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
#     elif command == "/exit":
#         os.kill(os.getpid(), signal.SIGTERM)
#         exit(0)
#
#     if (
#             refer_wav_path == "" or refer_wav_path is None
#             or prompt_text == "" or prompt_text is None
#             or prompt_language == "" or prompt_language is None
#     ):
#         refer_wav_path, prompt_text, prompt_language = (
#             default_refer_path,
#             default_refer_text,
#             default_refer_language,
#         )
#         if not has_preset:
#             raise HTTPException(status_code=400, detail="未指定参考音频且接口无预设")
#
#     with torch.no_grad():
#         gen = get_tts_wav(
#             refer_wav_path, prompt_text, prompt_language, text, text_language
#         )
#         sampling_rate, audio_data = next(gen)
#
#     wav = BytesIO()
#     sf.write(wav, audio_data, sampling_rate, format="wav")
#     wav.seek(0)
#
#     torch.cuda.empty_cache()
#     return StreamingResponse(wav, media_type="audio/wav")


# app = FastAPI()
#
#
# @app.post("/")
# async def tts_endpoint(request: Request):
#     json_post_raw = await request.json()
#     return handle(
#         json_post_raw.get("command"),
#         json_post_raw.get("refer_wav_path"),
#         json_post_raw.get("prompt_text"),
#         json_post_raw.get("prompt_language"),
#         json_post_raw.get("text"),
#         json_post_raw.get("text_language"),
#     )
#
#
# @app.get("/")
# async def tts_endpoint(
#         command: str = None,
#         refer_wav_path: str = None,
#         prompt_text: str = None,
#         prompt_language: str = None,
#         text: str = None,
#         text_language: str = None,
# ):
#     return handle(command, refer_wav_path, prompt_text, prompt_language, text, text_language)
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host=host, port=port, workers=1)
