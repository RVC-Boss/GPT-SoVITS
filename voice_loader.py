import config
import sys,os
import gradio as gr
import torch
import numpy as np
import librosa,torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto

hps = None
ssl_model = None
vq_model = None
t2s_model = None
is_half = config.is_half
hz = 50
max_sec = None
top_k = None

#后期可能将这里个path分离成变量
bert_path = config.bert_path
cnhubert_base_path = config.cnhubert_path
cnhubert.cnhubert_base_path = cnhubert_base_path
device = "cuda" #不确定能否支持cpu,先预留

tokenizer = None
bert_model = None

i18n = I18nAuto()
cwd = os.getcwd()
sys.path.append(cwd)

SUPPORT_LANGUAGE = [i18n("中文"),i18n("英文"),i18n("日文")]

dict_language={
    i18n("中文"):"zh",
    i18n("英文"):"en",
    i18n("日文"):"ja"
}

def read_model_path(model_type):
    model_list = []
    if model_type == config.MODEL_TYPE_GPT:
        folder_path = os.path.join(cwd,config.MODEL_FOLDER_PATH_GPT)
        file_type = ".ckpt"
    elif model_type == config.MODEL_TYPE_SOVITS:
        folder_path = os.path.join(cwd,config.MODEL_FOLDER_PATH_SOVITS)
        file_type = ".pth"
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(file_type):
                file_path = os.path.join(root, file_name)
                model_list.append((file_name,file_path))
    return model_list

def refresh_model_list():
    gpt_choices = read_model_path(config.MODEL_TYPE_GPT)
    sovits_choices = read_model_path(config.MODEL_TYPE_SOVITS)
    return gr.Dropdown(choices=sorted(gpt_choices),value=gpt_choices[0]if len(gpt_choices)>0  else "",interactive=True),gr.Dropdown(choices=sorted(sovits_choices),value=sovits_choices[0]if len(sovits_choices)>0  else None,interactive=True)

def get_bert_feature(text, word2ph):
    global tokenizer,bert_model
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

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
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
    return spec

def load_models(sovits_path, gpt_path):
    global tokenizer,bert_model,hps,ssl_model,vq_model,t2s_model,is_half,hz,max_sec,top_k
    print(f"SoVITS model path: {sovits_path}")
    print(f"GPT model path: {gpt_path}")

    if sovits_path is None or gpt_path is None:
        print("Choose both of two models before loading")
        return "请正确选择两个模型",gr.Button(interactive=False)

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    dict_s2=torch.load(sovits_path,map_location="cpu")
    hps=dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"

    dict_s1 = torch.load(gpt_path, map_location="cpu")
    dict_s1_config = dict_s1["config"]
    max_sec = dict_s1_config["data"]["max_sec"]
    top_k=dict_s1_config["inference"]["top_k"]

    ssl_model = cnhubert.get_model()
    if is_half == True:
        ssl_model = ssl_model.half().to(device)
    else:
        ssl_model = ssl_model.to(device)

    vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
    )
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))

    t2s_model = Text2SemanticLightningModule(dict_s1_config, "ojbk", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    #加载模型成功
    return "模型加载成功",gr.Button(interactive=True)
    

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language):
    global hps,ssl_model,vq_model,t2s_model,is_half,hz,max_sec,top_k
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k=torch.cat([wav16k,zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
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
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phones2 = cleaned_text_to_sequence(phones2)
        if prompt_language == "zh":
            bert1 = get_bert_feature(norm_text1, word2ph1).to(device)
        else:
            bert1 = torch.zeros(
                (1024, len(phones1)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
        if text_language == "zh":
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
                top_k=top_k,
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )

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
}  # 不考虑省略号


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


def start_webui():

    ngpu = torch.cuda.device_count()
    gpu_list = []
    for i in range(ngpu):
        gpu_list.append((torch.cuda.get_device_name(i),i))
    print(gpu_list)

    gpt_choices = read_model_path(config.MODEL_TYPE_GPT)
    sovits_choices = read_model_path(config.MODEL_TYPE_SOVITS)

    with gr.Blocks() as demo:
        with gr.Row():
            message_text = gr.Textbox("信息",interactive=False)
            with gr.Accordion(label="设备"):
                with gr.Row():
                    cuda_device_index = gr.Dropdown(choices=gpu_list,value=0 if len(gpu_list)>0 else None,label="CUDA设备",interactive=True)
        with gr.Accordion(label="模型"):
            with gr.Row():
                gpt_dropdown = gr.Dropdown(choices=sorted(gpt_choices),value=gpt_choices[0][1]if len(gpt_choices)>0  else None,label="选择GPT模型",interactive=True)
                sovits_dropdown = gr.Dropdown(choices=sorted(sovits_choices),value=sovits_choices[0][1]if len(sovits_choices)>0  else None,label="选择SoVITS模型",interactive=True)
                with gr.Row():
                    model_load_button = gr.Button("加载模型",variant="primary")
                    model_refresh_button = gr.Button("刷新模型", variant="secondary")
        with gr.Accordion(label="参考"):
            with gr.Group():
                with gr.Row():
                    with gr.Row():
                        ref_wav_path = gr.Audio(label="参考音频", type="filepath", scale=3)
                        ref_language = gr.Dropdown(choices=SUPPORT_LANGUAGE,value=i18n("中文"),label="参考语种",interactive=True,min_width=50, scale=1)
                    ref_text = gr.TextArea(label="参考文本",scale=1)
        with gr.Row():
            output_language = gr.Dropdown(choices=SUPPORT_LANGUAGE,value=i18n("中文"),label="合成语种",interactive=True, scale=2)
            preprocess_output_text_button = gr.Button("合成文本预处理",variant="primary",scale=3)
            inference_button = gr.Button(i18n("合成语音"), interactive=False,variant="primary")
        output_text = gr.TextArea(label="合成文本",interactive=True)
        output_audio = gr.Audio(label="输出结果")
        model_load_button.click(load_models,[gpt_dropdown,sovits_dropdown],[message_text,inference_button])
        model_refresh_button.click(refresh_model_list,[],[gpt_dropdown,sovits_dropdown])
        inference_button.click(
            get_tts_wav,
            [ref_wav_path, ref_text, ref_language, output_text, output_language],
            [output_audio],
        )
    demo.queue(max_size=1022).launch(server_port=2777)

if __name__ == "__main__":
    start_webui()





