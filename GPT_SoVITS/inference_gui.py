import os,re,logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb

if os.path.exists("./gweight.txt"):
    with open("./gweight.txt", 'r',encoding="utf-8") as file:
        gweight_data = file.read()
        gpt_path = os.environ.get(
    "gpt_path", gweight_data)
else:
    gpt_path = os.environ.get(
    "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

if os.path.exists("./sweight.txt"):
    with open("./sweight.txt", 'r',encoding="utf-8") as file:
        sweight_data = file.read()
        sovits_path = os.environ.get("sovits_path", sweight_data)
else:
    sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")
# gpt_path = os.environ.get(
#     "gpt_path", "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
# )
# sovits_path = os.environ.get("sovits_path", "pretrained_models/s2G488k.pth")
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share=eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True"))
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa,torch
from feature_extractor import cnhubert
cnhubert.cnhubert_base_path=cnhubert_base_path

import sys
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QWidget, QFileDialog, QStatusBar, QComboBox
import soundfile as sf

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

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

def change_sovits_weights(sovits_path):
    global vq_model,hps
    dict_s2=torch.load(sovits_path,map_location="cpu")
    hps=dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if("pretrained"not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt","w",encoding="utf-8")as f:f.write(sovits_path)
change_sovits_weights(sovits_path)

def change_gpt_weights(gpt_path):
    global hz,max_sec,t2s_model,config
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
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt","w",encoding="utf-8")as f:f.write(gpt_path)
change_gpt_weights(gpt_path)

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


dict_language={
    i18n("中文"):"zh",
    i18n("英文"):"en",
    i18n("日文"):"ja"
}


def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z. ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)

    return textlist, langlist


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)

    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


def nonen_clean_text_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "en" or "ja":
            pass
        else:
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text


def nonen_get_bert_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    print(textlist)
    print(langlist)
    bert_list = []
    for i in range(len(textlist)):
        text = textlist[i]
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(text, lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)

    return bert

splits = {"，","。","？","！",",",".","?","!","~",":","：","—","…",}
def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language,how_to_cut=i18n("不切")):
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    if(prompt_text[-1]not in splits):prompt_text+="。"if prompt_text!="en"else "."
    text = text.strip("\n")
    if(len(get_first(text))<4):text+="。"if text!="en"else "."
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if(wav16k.shape[0]>160000 or wav16k.shape[0]<48000):
            raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
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

    if prompt_language == "en":
        phones1, word2ph1, norm_text1 = clean_text_inf(prompt_text, prompt_language)
    else:
        phones1, word2ph1, norm_text1 = nonen_clean_text_inf(prompt_text, prompt_language)
    if(how_to_cut==i18n("凑四句一切")):text=cut1(text)
    elif(how_to_cut==i18n("凑50字一切")):text=cut2(text)
    elif(how_to_cut==i18n("按中文句号。切")):text=cut3(text)
    elif(how_to_cut==i18n("按英文句号.切")):text=cut4(text)
    text = text.replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n")
    if(text[-1]not in splits):text+="。"if text_language!="en"else "."
    texts=text.split("\n")
    audio_opt = []
    if prompt_language == "en":
        bert1 = get_bert_inf(phones1, word2ph1, norm_text1, prompt_language)
    else:
        bert1 = nonen_get_bert_inf(prompt_text, prompt_language)
    
    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if text_language == "en":
            phones2, word2ph2, norm_text2 = clean_text_inf(text, text_language)
        else:
            phones2, word2ph2, norm_text2 = nonen_clean_text_inf(text, text_language)

        if text_language == "en":
            bert2 = get_bert_inf(phones2, word2ph2, norm_text2, text_language)
        else:
            bert2 = nonen_get_bert_inf(text, text_language)

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
                top_k=config["inference"]["top_k"],
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
    if len(opts)>1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s。" % item for item in inp.strip("。").split("。")])
def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s." % item for item in inp.strip(".").split(".")])

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names,key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names,key=custom_sort_key), "__type__": "update"}

pretrained_sovits_name="GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root="SoVITS_weights"
GPT_weight_root="GPT_weights"
os.makedirs(SoVITS_weight_root,exist_ok=True)
os.makedirs(GPT_weight_root,exist_ok=True)

def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):SoVITS_names.append("%s/%s"%(SoVITS_weight_root,name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s"%(GPT_weight_root,name))
    return SoVITS_names,GPT_names
SoVITS_names,GPT_names = get_weights_names()


class GPTSoVITSGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('GPT-SoVITS GUI')
        self.setGeometry(800, 450, 950, 850)

        self.setStyleSheet("""
            QWidget {
                background-color: #a3d3b1; 
            }

            QTabWidget::pane {
                background-color: #a3d3b1;  
            }

            QTabWidget::tab-bar {
                alignment: left;
            }

            QTabBar::tab {
                background: #8da4bf; 
                color: #ffffff;  
                padding: 8px;
            }

            QTabBar::tab:selected {
                background: #2a3f54; 
            }

            QLabel {
                color: #000000;  
            }

            QPushButton {
                background-color: #4CAF50; 
                color: white;  
                padding: 8px;
                border: 1px solid #4CAF50;
                border-radius: 4px;
            }

            QPushButton:hover {
                background-color: #45a049;  
                border: 1px solid #45a049;
                box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.1);
            }
        """)    

        license_text = (
        "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. "
        "如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        license_label = QLabel(license_text)
        license_label.setWordWrap(True)

        self.GPT_model_label = QLabel("选择GPT模型:")
        self.GPT_model_input = QLineEdit()
        self.GPT_model_input.setPlaceholderText("拖拽或选择文件")
        self.GPT_model_input.setReadOnly(True)
        self.GPT_model_button = QPushButton("选择GPT模型文件")
        self.GPT_model_button.clicked.connect(self.select_GPT_model)

        self.SoVITS_model_label = QLabel("选择SoVITS模型:")
        self.SoVITS_model_input = QLineEdit()
        self.SoVITS_model_input.setPlaceholderText("拖拽或选择文件")
        self.SoVITS_model_input.setReadOnly(True)
        self.SoVITS_model_button = QPushButton("选择SoVITS模型文件")
        self.SoVITS_model_button.clicked.connect(self.select_SoVITS_model)

        self.ref_audio_label = QLabel("上传参考音频:")
        self.ref_audio_input = QLineEdit()
        self.ref_audio_input.setPlaceholderText("拖拽或选择文件")
        self.ref_audio_input.setReadOnly(True)
        self.ref_audio_button = QPushButton("选择音频文件")
        self.ref_audio_button.clicked.connect(self.select_ref_audio)

        self.ref_text_label = QLabel("参考音频文本:")
        self.ref_text_input = QLineEdit()
        self.ref_text_input.setPlaceholderText("拖拽或选择文件")
        self.ref_text_input.setReadOnly(True)
        self.ref_text_button = QPushButton("上传文本")
        self.ref_text_button.clicked.connect(self.upload_ref_text)

        self.language_label = QLabel("参考音频语言:")
        self.language_combobox = QComboBox()
        self.language_combobox.addItems(["中文", "英文", "日文"])

        self.target_text_label = QLabel("合成目标文本:")
        self.target_text_input = QLineEdit()
        self.target_text_input.setPlaceholderText("拖拽或选择文件")
        self.target_text_input.setReadOnly(True)
        self.target_text_button = QPushButton("上传文本")
        self.target_text_button.clicked.connect(self.upload_target_text)

        self.language_label_02 = QLabel("合成音频语言:")
        self.language_combobox_02 = QComboBox()
        self.language_combobox_02.addItems(["中文", "英文", "日文"])

        self.output_label = QLabel("输出音频路径:")
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("拖拽或选择文件")
        self.output_input.setReadOnly(True)
        self.output_button = QPushButton("选择文件夹")
        self.output_button.clicked.connect(self.select_output_path)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        self.add_drag_drop_events([
            self.GPT_model_input,
            self.SoVITS_model_input,
            self.ref_audio_input,
            self.ref_text_input,
            self.target_text_input,
            self.output_input,
        ])

        self.synthesize_button = QPushButton("合成")
        self.synthesize_button.clicked.connect(self.synthesize)

        self.status_bar = QStatusBar()

        main_layout = QVBoxLayout()

        input_layout = QGridLayout()
        input_layout.setSpacing(10) 

        self.setLayout(input_layout)

        input_layout.addWidget(license_label, 0, 0, 1, 3)

        input_layout.addWidget(self.GPT_model_label, 1, 0)
        input_layout.addWidget(self.GPT_model_input, 2, 0, 1, 2)
        input_layout.addWidget(self.GPT_model_button, 2, 2)

        input_layout.addWidget(self.SoVITS_model_label, 3, 0)
        input_layout.addWidget(self.SoVITS_model_input, 4, 0, 1, 2)
        input_layout.addWidget(self.SoVITS_model_button, 4, 2)

        input_layout.addWidget(self.ref_audio_label, 5, 0)
        input_layout.addWidget(self.ref_audio_input, 6, 0, 1, 2)
        input_layout.addWidget(self.ref_audio_button, 6, 2)

        input_layout.addWidget(self.language_label, 7, 0)
        input_layout.addWidget(self.language_combobox, 8, 0, 1, 1)
        input_layout.addWidget(self.ref_text_label, 9, 0)
        input_layout.addWidget(self.ref_text_input, 10, 0, 1, 2)
        input_layout.addWidget(self.ref_text_button, 10, 2)

        input_layout.addWidget(self.language_label_02, 11, 0)
        input_layout.addWidget(self.language_combobox_02, 12, 0, 1, 1)
        input_layout.addWidget(self.target_text_label, 13, 0)
        input_layout.addWidget(self.target_text_input, 14, 0, 1, 2)
        input_layout.addWidget(self.target_text_button, 14, 2)
        
        input_layout.addWidget(self.output_label, 15, 0)
        input_layout.addWidget(self.output_input, 16, 0, 1, 2)
        input_layout.addWidget(self.output_button, 16, 2)
        
        main_layout.addLayout(input_layout)

        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_text)
        main_layout.addLayout(output_layout)

        main_layout.addWidget(self.synthesize_button)

        main_layout.addWidget(self.status_bar)

        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
          
            if len(file_paths) == 1:
                self.update_ref_audio(file_paths[0])
                self.update_input_paths(self.ref_audio_input, file_paths[0])
            else:
                self.update_ref_audio(", ".join(file_paths))

    def add_drag_drop_events(self, widgets):
        for widget in widgets:
            widget.setAcceptDrops(True)
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.DragEnter:
            mime_data = event.mimeData()
            if mime_data.hasUrls():
                event.acceptProposedAction()
              
        elif event.type() == QEvent.Drop:
            mime_data = event.mimeData()
            if mime_data.hasUrls():
                file_paths = [url.toLocalFile() for url in mime_data.urls()]
                if len(file_paths) == 1:
                    self.update_input_paths(obj, file_paths[0])
                else:
                    self.update_input_paths(obj, ", ".join(file_paths))
                event.acceptProposedAction()

        return super().eventFilter(obj, event)
    
    def select_GPT_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择GPT模型文件", "", "GPT Files (*.ckpt)")
        if file_path:
            self.GPT_model_input.setText(file_path)

    def select_SoVITS_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择SoVITS模型文件", "", "SoVITS Files (*.pth)")
        if file_path:
            self.SoVITS_model_input.setText(file_path)

    def select_ref_audio(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.ShowDirsOnly

        file_dialog = QFileDialog()
        file_dialog.setOptions(options)

        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter("Audio Files (*.wav *.mp3)")

        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
          
            if len(file_paths) == 1:
                self.update_ref_audio(file_paths[0])
                self.update_input_paths(self.ref_audio_input, file_paths[0])
            else:
                self.update_ref_audio(", ".join(file_paths))

    def upload_ref_text(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文本文件", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.ref_text_input.setText(content)
                self.update_input_paths(self.ref_text_input, file_path)

    def upload_target_text(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文本文件", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.target_text_input.setText(content)
                self.update_input_paths(self.target_text_input, file_path)

    def select_output_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.ShowDirsOnly

        folder_dialog = QFileDialog()
        folder_dialog.setOptions(options)
        folder_dialog.setFileMode(QFileDialog.Directory)

        if folder_dialog.exec_():
            folder_path = folder_dialog.selectedFiles()[0]
            self.output_input.setText(folder_path)

    def update_ref_audio(self, file_path):
        self.ref_audio_input.setText(file_path)

    def update_input_paths(self, input_box, file_path):
        input_box.setText(file_path)

    def synthesize(self):
        GPT_model_path = self.GPT_model_input.text()
        SoVITS_model_path = self.SoVITS_model_input.text()
        ref_audio_path = self.ref_audio_input.text()
        language_combobox = self.language_combobox.currentText()
        language_combobox = i18n(language_combobox)
        ref_text = self.ref_text_input.text()
        language_combobox_02 = self.language_combobox_02.currentText()
        language_combobox_02 = i18n(language_combobox_02)
        target_text = self.target_text_input.text()
        output_path = self.output_input.text()

        change_gpt_weights(gpt_path=GPT_model_path)
        change_sovits_weights(sovits_path=SoVITS_model_path)

        synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path, 
                                       prompt_text=ref_text, 
                                       prompt_language=language_combobox, 
                                       text=target_text, 
                                       text_language=language_combobox_02)
        
        result_list = list(synthesis_result)

        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            output_wav_path = os.path.join(output_path, "output.wav") 
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)

            result = "Audio saved to " + output_wav_path

        self.status_bar.showMessage("合成完成！输出路径：" + output_wav_path, 5000)
        self.output_text.append("处理结果：\n" + result)

def main():
    app = QApplication(sys.argv)
    mainWin = GPTSoVITSGUI()
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
