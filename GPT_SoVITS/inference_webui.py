import os
import numpy as np
import librosa,torch
from feature_extractor import cnhubert

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from transformers import AutoModelForMaskedLM, AutoTokenizer
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio


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


class Inference:
    def __init__(self, is_half, GPT_weight_root, SoVITS_weight_root):
        self.n_semantic = 1024
        self.model_loaded = False
        self.is_half = is_half
        self.GPT_weight_root = GPT_weight_root
        self.SoVITS_weight_root = SoVITS_weight_root


    def update_envs(self, gpt_path, sovits_path, cnhubert_base_path, bert_path):
        self.gpt_path =  os.path.join(self.GPT_weight_root, gpt_path)
        self.sovits_path = os.path.join(self.SoVITS_weight_root, sovits_path)
        self.cnhubert_base_path = cnhubert_base_path
        self.bert_path = bert_path

        cnhubert.cnhubert_base_path=cnhubert_base_path

        yield self.load_model()
    
    def load_model(self, device='cuda'):
        try:
            # Load bert model
            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)
            if self.is_half == True:
                self.bert_model = self.bert_model.half().to(device)
            else:
                self.bert_model = self.bert_model.to(device)

            # Load ssl model
            dict_s1 = torch.load(self.gpt_path, map_location="cpu")
            self.config = dict_s1["config"]
            self.ssl_model = cnhubert.get_model()
            if self.is_half == True:
                self.ssl_model = self.ssl_model.half().to(device)
            else:
                self.ssl_model = self.ssl_model.to(device)
            
            dict_s2=torch.load(self.sovits_path,map_location="cpu")
            self.hps=dict_s2["config"]
            self.hps = DictToAttrRecursive(self.hps)
            self.hps.model.semantic_frame_rate = "25hz"

            # Load vq model
            self.vq_model = SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model
            )
            if self.is_half == True:
                self.vq_model = self.vq_model.half().to(device)
            else:
                self.vq_model = self.vq_model.to(device)
            self.vq_model.eval()
            self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

            # Load t2s model 
            # t2s_model = Text2SemanticLightningModule.load_from_checkpoint(checkpoint_path=gpt_path, config=config, map_location="cpu")#########todo
            self.t2s_model = Text2SemanticLightningModule(self.config, "ojbk", is_train=False)
            self.t2s_model.load_state_dict(dict_s1["weight"])
            if self.is_half == True:
                self.t2s_model = self.t2s_model.half()
            self.t2s_model = self.t2s_model.to(device)
            self.t2s_model.eval()
            total = sum([param.nelement() for param in self.t2s_model.parameters()])
            print("Number of parameter: %.2fM" % (total / 1e6))

            self.model_loaded = True
            return '模型加载成功'
        except Exception as e:
            return f'模型加载失败：{e}'
    
    def unload_model(self):
        if self.model_loaded:
            try:
                del self.bert_model, self.ssl_model, self.hps, self.vq_model, self.t2s_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model_loaded = False
                yield '模型卸载成功'
            except Exception as e:
                yield f'模型卸载失败：{e}'
        else:
            yield '模型未加载'

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)  #####输入是long不用管精度问题，精度随bert_model
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        # if(is_half==True):phone_level_feature=phone_level_feature.half()
        return phone_level_feature.T


    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language):
        if not self.model_loaded:
            return 
        hz = 50
        dict_language = {"中文": "zh", "英文": "en", "日文": "ja"}
        t0 = ttime()
        prompt_text = prompt_text.strip("\n")
        prompt_language, text = prompt_language, text.strip("\n")
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)  # 派蒙
            wav16k = torch.from_numpy(wav16k)
            if self.is_half == True:
                wav16k = wav16k.half().to(self.device)
            else:
                wav16k = wav16k.to(self.device)
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
        t1 = ttime()
        prompt_language = dict_language[prompt_language]
        text_language = dict_language[text_language]
        phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
        phones1 = cleaned_text_to_sequence(phones1)
        texts = text.split("\n")
        audio_opt = []
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half == True else np.float32,
        )
        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            phones2, word2ph2, norm_text2 = clean_text(text, text_language)
            phones2 = cleaned_text_to_sequence(phones2)
            if prompt_language == "zh":
                bert1 = self.get_bert_feature(norm_text1, word2ph1).to(self.device)
            else:
                bert1 = torch.zeros(
                    (1024, len(phones1)),
                    dtype=torch.float16 if self.is_half == True else torch.float32,
                ).to(self.device)
            if text_language == "zh":
                bert2 = self.get_bert_feature(norm_text2, word2ph2).to(self.device)
            else:
                bert2 = torch.zeros((1024, len(phones2))).to(bert1)
            bert = torch.cat([bert1, bert2], 1)

            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = prompt_semantic.unsqueeze(0).to(self.device)
            t2 = ttime()
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=self.config["inference"]["top_k"],
                    early_stop_num=hz * self.config["data"]["max_sec"]
                )
            t3 = ttime()
            # print(pred_semantic.shape,idx)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(
                0
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = get_spepc(self.hps, ref_wav_path)  # .to(device)
            if self.is_half == True:
                refer = refer.half().to(self.device)
            else:
                refer = refer.to(self.device)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = (
                self.vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refer
                )
                .detach()
                .cpu()
                .numpy()[0, 0]
            )  ###试试重建不带上prompt部分
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            t4 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        yield self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
            np.int16
        )



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



def split(todo_text):
    splits = { "，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }  # 不考虑省略号
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
    split_idx = list(range(0, len(inps), 5))
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
        return [inp]
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
    if len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s。" % item for item in inp.strip("。").split("。")])

