import os
import re
import sys

from cmd_args import CmdArgs
from pyutils.logs import llog

current_project_dir = os.getcwd()
sys.path.append(current_project_dir)
sys.path.append("%s/GPT_SoVITS" % (current_project_dir))

import signal
import LangSegment
from time import time as ttime
import torch
import librosa
import soundfile as sf
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from feature_extractor import cnhubert
from io import BytesIO
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from pyutils.np_utils import load_audio
import subprocess
import config as global_config

g_config = global_config.Config()


class DefaultRefer:
  def __init__(self, path, text, language):
    self.path = args.default_refer_path
    self.text = args.default_refer_text
    self.language = args.default_refer_language

  def is_ready(self) -> bool:
    return is_not_empty(self.path, self.text, self.language)


def is_empty(*items):  # 任意一项不为空返回False
  for item in items:
    if item is not None and item != "":
      return False
  return True


def is_not_empty(*items):  # 任意一项为空返回False
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
  model_params_dict = vars(hps.model)
  vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **model_params_dict
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
  global hz, max_sec, t2s_model, config
  hz = 50
  dict_s1 = torch.load(gpt_path, map_location="cpu")
  config = dict_s1["config"]
  max_sec = config["data"]["max_sec"]
  t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
  t2s_model.load_state_dict(dict_s1["weight"])
  if is_half == True:
    t2s_model = t2s_model.half()
  t2s_model = t2s_model.to(device)
  t2s_model.eval()
  total = sum([param.nelement() for param in t2s_model.parameters()])
  llog.info("Number of parameter: %.2fM" % (total / 1e6))


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
  language = language.replace("all_", "")
  if language == "zh":
    bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
  else:
    bert = torch.zeros(
      (1024, len(phones)),
      dtype=torch.float16 if is_half == True else torch.float32,
    ).to(device)

  return bert


def get_phones_and_bert(text, language):
  if language in {"en", "all_zh", "all_ja"}:
    language = language.replace("all_", "")
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
        dtype=torch.float16 if is_half == True else torch.float32,
      ).to(device)
  elif language in {"zh", "ja", "auto"}:
    textlist = []
    langlist = []
    LangSegment.setfilters(["zh", "ja", "en", "ko"])
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
    # llog.info(textlist)
    # llog.info(langlist)
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

  return phones, bert.to(torch.float16 if is_half == True else torch.float32), norm_text


class DictToAttrRecursive:
  def __init__(self, input_dict):
    for key, value in input_dict.items():
      if isinstance(value, dict):
        # 如果值是字典，递归调用构造函数
        setattr(self, key, DictToAttrRecursive(value))
      else:
        setattr(self, key, value)


def get_spepc(hps, filename):
  audio = load_audio(filename, int(hps.data.sampling_rate))
  audio = torch.FloatTensor(audio)
  audio_norm = audio
  audio_norm = audio_norm.unsqueeze(0)
  spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                           hps.data.win_length, center=False)
  return spec


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
  data = np.frombuffer(audio_bytes.getvalue(), dtype=np.int16)
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
  punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
  if len(punc_list) > 0:
    punds = r"[" + "".join(punc_list) + r"]"
    text = text.strip("\n")
    items = re.split(f"({punds})", text)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items) % 2 == 1:
      mergeitems.append(items[-1])
    text = "\n".join(mergeitems)

  while "\n\n" in text:
    text = text.replace("\n\n", "\n")

  return text


def only_punc(text):
  return not any(t.isalnum() or t.isalpha() for t in text)


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
  prompt_language = dict_language[prompt_language.lower()]
  text_language = dict_language[text_language.lower()]
  phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)
  texts = text.split("\n")
  audio_bytes = BytesIO()

  for text in texts:
    # 简单防止纯符号引发参考音频泄露
    if only_punc(text):
      continue

    audio_opt = []
    phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
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
    audio_bytes = pack_audio(audio_bytes, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16),
                             hps.data.sampling_rate)
    # llog.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    if stream_mode == "normal":
      audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
      yield audio_chunk

  if not stream_mode == "normal":
    if media_type == "wav":
      audio_bytes = pack_wav(audio_bytes, hps.data.sampling_rate)
    yield audio_bytes.getvalue()


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

  llog.info(f"当前默认参考音频路径: {default_refer.path}")
  llog.info(f"当前默认参考音频文本: {default_refer.text}")
  llog.info(f"当前默认参考音频语种: {default_refer.language}")
  llog.info(f"is_ready: {default_refer.is_ready()}")

  return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


def handle(refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc):
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

  if cut_punc == None:
    text = cut_text(text, default_cut_punc)
  else:
    text = cut_text(text, cut_punc)

  return StreamingResponse(get_tts_wav(refer_wav_path, prompt_text, prompt_language, text, text_language),
                           media_type="audio/" + media_type)


# --------------------------------
# 初始化部分
# --------------------------------
dict_language = {
  "中文": "all_zh",
  "英文": "en",
  "日文": "all_ja",
  "中英混合": "zh",
  "日英混合": "ja",
  "多语种混合": "auto",  # 多语种启动切分识别语种
  "all_zh": "all_zh",
  "en": "en",
  "all_ja": "all_ja",
  "zh": "zh",
  "ja": "ja",
  "auto": "auto",
}

# 获取配置
cmd_args = CmdArgs()
args = cmd_args.get_args()
sovits_path = args.sovits_path
gpt_path = args.gpt_path
device = args.device

cnhubert_base_path = args.hubert_path
bert_path = args.bert_path
default_cut_punc = args.cut_punc

# 应用参数配置
default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)

# 模型路径检查
if sovits_path == "":
  sovits_path = g_config.pretrained_sovits_path
  llog.warn(f"未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
  gpt_path = g_config.pretrained_gpt_path
  llog.warn(f"未指定GPT模型路径, fallback后当前值: {gpt_path}")

# 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
if default_refer.path == "" or default_refer.text == "" or default_refer.language == "":
  default_refer.path, default_refer.text, default_refer.language = "", "", ""
  llog.info("未指定默认参考音频")
else:
  llog.info(f"默认参考音频路径: {default_refer.path}")
  llog.info(f"默认参考音频文本: {default_refer.text}")
  llog.info(f"默认参考音频语种: {default_refer.language}")

# 获取半精度
is_half = g_config.is_half
if args.full_precision:
  is_half = False
if args.half_precision:
  is_half = True
if args.full_precision and args.half_precision:
  is_half = g_config.is_half  # 炒饭fallback
llog.info(f"半精: {is_half}")

# 流式返回模式
if args.stream_mode.lower() in ["normal", "n"]:
  stream_mode = "normal"
  llog.info("流式返回已开启")
else:
  stream_mode = "close"

# 音频编码格式
if args.media_type.lower() in ["aac", "ogg"]:
  media_type = args.media_type.lower()
elif stream_mode == "close":
  media_type = "wav"
else:
  media_type = "ogg"
llog.info(f"编码格式: {media_type}")

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
