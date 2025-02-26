#!/usr/bin/env python
# coding=utf-8
import gradio as gr
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
import librosa
from feature_extractor import cnhubert
from GPT_SoVITS.module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import chinese, cleaned_text_to_sequence
from text.cleaner import clean_text
from text.LangSegmenter import LangSegmenter
from time import time as ttime
from module.mel_processing import spectrogram_torch, spec_to_mel_torch
from tools.my_utils import load_audio
import torch, torchaudio
import traceback
import os, re

# 模型路径
gpt_path = 'GPT_weights_v2/amiya-e50.ckpt'
sovits_path = 'SoVITS_weights_v2/amiya_e25_s950.pth'
cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
cnhubert.cnhubert_base_path = cnhubert_base_path

# 参考音频相关配置
REFERENCE_AUDIO = "/Users/baysonfox/Desktop/amiya-chatbot/reference.mp3"
REFERENCE_TEXT = "博士，休息好了吗？还觉得累的话，不用勉强的。有我在呢。"
INF_REFS = [os.path.join("/Users/baysonfox/Desktop/amiya-chatbot/references", f) for f in os.listdir("/Users/baysonfox/Desktop/amiya-chatbot/references")]

# 模型相关设置
DEVICE = 'cpu'
dict_language = {
    "中文": "all_zh",#全部按中文识别
    "英文": "en",#全部按英文识别#######不变
    "日文": "all_ja",#全部按日文识别
    "中英混合": "zh",#按中英混合识别####不变
    "日英混合": "ja",#按日英混合识别####不变
    "多语种混合": "auto",#多语种启动切分识别语种
}
os.environ["TOKENIZERS_PARALLELISM"] = "False"

DTYPE = torch.float32
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
ssl_model = cnhubert.get_model()

# Accelerated Inference
tokenizer = torch.compile(tokenizer)
bert_model = torch.compile(bert_model)
ssl_model = torch.compile(ssl_model)

# 标点符号
PUNCTUATION = {'!', '?', '…', ',', '.', '-', " "}
# 中文标点符号
SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

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

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(DEVICE)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

resample_transform_dict={}
def resample(audio_tensor, sr0):
    global resample_transform_dict
    if sr0 not in resample_transform_dict:
        resample_transform_dict[sr0] = torchaudio.transforms.Resample(
            sr0, 24000
        ).to(DEVICE)
    return resample_transform_dict[sr0](audio_tensor)

def change_sovits_weights(prompt_language=None,text_language=None):
    global vq_model, hps, version, model_version, dict_language
    model_version = version = "v2"

    if prompt_language is not None and text_language is not None:
        prompt_text_update, prompt_language_update = {'__type__':'update'},  {'__type__':'update', 'value': "all_zh"}

        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {'__type__':'update'}, {'__type__':'update', 'value':text_language}
        else:
            text_update = {'__type__':'update', 'value':''}
            text_language_update = {'__type__':'update', 'value':"中文"}
        yield {'__type__':'update', 'choices':list(dict_language.keys())}, {'__type__':'update', 'choices':list(dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update,{"__type__": "update", "visible": False},{"__type__": "update", "visible": False},{"__type__": "update", "value": False,"interactive": False}

    dict_s2 = torch.load(sovits_path, map_location="cpu")

    hps = DictToAttrRecursive(dict_s2["config"])

    hps.model.semantic_frame_rate = "25hz"
    hps.model.version = "v2"
    version = hps.model.version
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    model_version = version

    vq_model = vq_model.to(DEVICE)
    print("loading sovits_%s" % model_version,vq_model.load_state_dict(
        dict_s2["weight"],
        strict=False))
    vq_model.eval()
    vq_model = torch.compile(vq_model)

def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_model = t2s_model.to(DEVICE)
    t2s_model.eval()
    t2s_model = torch.compile(t2s_model)

def get_spepc(hps, filename):
    print("hps samplingrate", hps.data.sampling_rate)
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx = audio.abs().max()
    if(maxx > 1):
        audio /= min(2, maxx.item())
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

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    print("phones: ", phones)
    print("word2ph: ", word2ph)
    print("norm_text: ", norm_text)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(DEVICE)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float32
        ).to(DEVICE)

    return bert

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in SPLITS) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def get_phones_and_bert(text,language,version,final=False):
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_","")
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(DEVICE)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32
            ).to(DEVICE)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
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
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)

    return phones,bert.to(DTYPE),norm_text

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    spec=spectrogram_torch(y,n_fft,sampling_rate,hop_size,win_size,center)
    mel=spec_to_mel_torch(spec,n_fft,num_mels,sampling_rate,fmin,fmax)
    return mel
mel_fn_args = {
    "n_fft": 1024,
    "win_size": 1024,
    "hop_size": 256,
    "num_mels": 100,
    "sampling_rate": 24000,
    "fmin": 0,
    "fmax": None,
    "center": False
}

spec_min = -12
spec_max = 2
def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1
def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min
mel_fn=lambda x: mel_spectrogram(x, **mel_fn_args)

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
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in SPLITS:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in SPLITS:
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
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
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
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)

def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return  "\n".join(opts)

def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r'(?<!\d)\.(?!\d)', inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)

def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
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
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def process_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError("请输入有效文本")
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

cache = {}
def get_tts_wav(text,
    text_language,
    how_to_cut="不切",
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    speed=1,
    if_freeze=False,
    sample_steps=8):

    global cache
    prompt_text = REFERENCE_TEXT
    prompt_language = "中文"
    ref_wav_path = REFERENCE_AUDIO
    inp_refs = INF_REFS
    t = []
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    # 去除prompt_text的换行
    prompt_text = prompt_text.strip("\n")
    # 手动添加标点符号
    if (prompt_text[-1] not in SPLITS): prompt_text += "。" if prompt_language != "en" else "."
    print("实际输入的参考文本:", prompt_text)
    text = text.strip("\n")

    print("实际输入的目标文本:", text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float32
    )

    with torch.no_grad():
        # 参考音频和sampling rate，numpy格式
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        # numpy -> torch
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        wav16k = wav16k.to(DEVICE)
        zero_wav_torch = zero_wav_torch.to(DEVICE)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(DEVICE)

    t1 = ttime()
    t.append(t1-t0)

    if how_to_cut != "不切":
        cut_map = {
            "凑四句一切": cut1,
            "凑50字一切": cut2,
            "按中文句号。切": cut3,
            "按英文句号.切": cut4,
            "按标点符号切": cut5
        }
        cut_map[how_to_cut](text)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    print("实际输入的目标文本(切句后):", text)

    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language, version)

    for i_text,text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in SPLITS): text += "。" if text_language != "en" else "."
        print("实际输入的目标文本(每句):", text)
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language, version)
        print("前端处理后的文本(每句):", norm_text2)
        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(DEVICE).unsqueeze(0)

        bert = bert.to(DEVICE).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(DEVICE)

        t2 = ttime()
        if(i_text in cache and if_freeze==True):pred_semantic=cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text]=pred_semantic
        t3 = ttime()
        refers=[]
        if(inp_refs):
            for path in inp_refs:
                try:
                    refer = get_spepc(hps, path).to(DTYPE).to(DEVICE)
                    refers.append(refer)
                except:
                    traceback.print_exc()
            if(len(refers)==0):refers = [get_spepc(hps, ref_wav_path).to(DTYPE).to(DEVICE)]
            audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(DEVICE).unsqueeze(0), refers,speed=speed).detach().cpu().numpy()[0, 0])

        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
        t.extend([t2 - t1,t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" %
            (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3]))
            )
    sr=hps.data.sampling_rate if model_version!="v3"else 24000
    yield sr, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

with gr.Blocks(title="GPT-SoVITS WebUI") as app:

    try:next(change_sovits_weights(sovits_path))
    except:pass
    change_gpt_weights(gpt_path) # 初始化GPT模型

    gr.Markdown(value="<h1 style='color: #2b5278; font-size: 24px; text-align: center;'>大概可能也许是阿米娅的声音（</h1>")
    with gr.Row() as main_row:
        with gr.Column(scale=7) as text_column:
            text = gr.Textbox(
                label="需要合成的文本",
                value="",
                lines=13,
                max_lines=13
            )

            with gr.Row():
                inference_button = gr.Button("合成语音", variant="primary", size='lg')
                output = gr.Audio(label="输出的语音")

        with gr.Column(scale=5) as control_column:
            text_language = gr.Dropdown(
                label="需要合成的语种。限制范围越小判别效果越好。",
                choices=list(dict_language.keys()),
                value="中文"
            )

            how_to_cut = gr.Dropdown(
                label="怎么切",
                choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"],
                value="凑四句一切"
            )

            gr.Markdown(value="语速调整，高为更快")

            if_freeze = gr.Checkbox(
                label="是否直接对上次合成结果调整语速和音色。防止随机性。",
                value=False
            )

            speed = gr.Slider(
                minimum=0.6,
                maximum=1.65,
                step=0.05,
                label="语速",
                value=1
            )

            gr.Markdown("GPT采样参数(无参考文本时不要太低。不懂就用默认)：")

            top_k = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                label="top_k",
                value=15
            )

            top_p = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.05,
                label="top_p",
                value=1
            )

            temperature = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.05,
                label="temperature",
                value=1
            )

    inference_button.click(
        get_tts_wav,
        [text, text_language, how_to_cut, top_k, top_p, temperature, speed, if_freeze],
        [output],
    )



if __name__ == '__main__':
    app.queue().launch(#concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=False,
        share=False,
        server_port=9872,
        quiet=True,
    )
