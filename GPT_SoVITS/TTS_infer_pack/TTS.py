import os, sys

import ffmpeg
now_dir = os.getcwd()
sys.path.append(now_dir)
import os
from typing import Generator, List, Union
import numpy as np
import torch
import yaml
from transformers import AutoModelForMaskedLM, AutoTokenizer

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from feature_extractor.cnhubert import CNHubert
from module.models import SynthesizerTrn
import librosa
from time import time as ttime
from tools.i18n.i18n import I18nAuto
from my_utils import load_audio
from module.mel_processing import spectrogram_torch
from .text_segmentation_method import splits
from .TextPreprocessor import TextPreprocessor
i18n = I18nAuto()

# tts_infer.yaml
"""
default:
  device: cpu
  is_half: false
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/s2G488k.pth
  
custom:
  device: cuda
  is_half: true
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/s2G488k.pth
  


"""



class TTS_Config:
    def __init__(self, configs: Union[dict, str]):
        configs_base_path:str = "GPT_SoVITS/configs/"
        os.makedirs(configs_base_path, exist_ok=True)
        self.configs_path:str = os.path.join(configs_base_path, "tts_infer.yaml")
        if isinstance(configs, str):
            self.configs_path = configs
            configs:dict = self._load_configs(configs)
            
        # assert isinstance(configs, dict)
        self.default_configs:dict = configs.get("default", None)
        if self.default_configs is None:
            self.default_configs={
                "device": "cpu",
                "is_half": False,
                "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                "vits_weights_path": "GPT_SoVITS/pretrained_models/s2G488k.pth",
                "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
            }
        self.configs:dict = configs.get("custom", self.default_configs)
        
        self.device = self.configs.get("device") 
        self.is_half = self.configs.get("is_half")
        self.t2s_weights_path = self.configs.get("t2s_weights_path")
        self.vits_weights_path = self.configs.get("vits_weights_path")
        self.bert_base_path = self.configs.get("bert_base_path")
        self.cnhuhbert_base_path = self.configs.get("cnhuhbert_base_path")
        
        
        self.max_sec = None
        self.hz:int = 50
        self.semantic_frame_rate:str = "25hz"
        self.segment_size:int = 20480
        self.filter_length:int = 2048
        self.sampling_rate:int = 32000
        self.hop_length:int = 640
        self.win_length:int = 2048
        self.n_speakers:int = 300
        
        self.langauges:list = ["auto", "en", "zh", "ja",  "all_zh", "all_ja"]
        print(self)
            
    def _load_configs(self, configs_path: str)->dict:
        with open(configs_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
    
        return configs
    

    def save_configs(self, configs_path:str=None)->None:
        configs={
            "default": {
                "device": "cpu",
                "is_half": False,
                "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                "vits_weights_path": "GPT_SoVITS/pretrained_models/s2G488k.pth",
                "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
            },
            "custom": {
                "device": str(self.device),
                "is_half": self.is_half,
                "t2s_weights_path": self.t2s_weights_path,
                "vits_weights_path": self.vits_weights_path,
                "bert_base_path": self.bert_base_path,
                "cnhuhbert_base_path": self.cnhuhbert_base_path
            }
        }
        if configs_path is None:
            configs_path = self.configs_path
        with open(configs_path, 'w') as f:
            yaml.dump(configs, f)
            
            
    def __str__(self):
        string = "----------------TTS Config--------------\n"
        string += "device: {}\n".format(self.device)
        string += "is_half: {}\n".format(self.is_half)
        string += "bert_base_path: {}\n".format(self.bert_base_path)
        string += "t2s_weights_path: {}\n".format(self.t2s_weights_path)
        string += "vits_weights_path: {}\n".format(self.vits_weights_path)
        string += "cnhuhbert_base_path: {}\n".format(self.cnhuhbert_base_path)
        string += "----------------------------------------\n"
        return string


class TTS:
    def __init__(self, configs: Union[dict, str, TTS_Config]):
        if isinstance(configs, TTS_Config):
            self.configs = configs
        else:
            self.configs:TTS_Config = TTS_Config(configs)
        
        self.t2s_model:Text2SemanticLightningModule = None
        self.vits_model:SynthesizerTrn = None
        self.bert_tokenizer:AutoTokenizer = None
        self.bert_model:AutoModelForMaskedLM = None
        self.cnhuhbert_model:CNHubert = None
        
        self._init_models()
        
        self.text_preprocessor:TextPreprocessor = \
                            TextPreprocessor(self.bert_model, 
                                            self.bert_tokenizer, 
                                            self.configs.device)
        
        
        self.prompt_cache:dict = {
            "ref_audio_path":None,
            "prompt_semantic":None,
            "refer_spepc":None,
            "prompt_text":None,
            "prompt_lang":None,
            "phones":None,
            "bert_features":None,
            "norm_text":None,
        }

    def _init_models(self,):
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_vits_weights(self.configs.vits_weights_path)
        self.init_bert_weights(self.configs.bert_base_path)
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)
        
        
        
    def init_cnhuhbert_weights(self, base_path: str):
        self.cnhuhbert_model = CNHubert(base_path)
        self.cnhuhbert_model.eval()
        if self.configs.is_half == True:
            self.cnhuhbert_model = self.cnhuhbert_model.half()
        self.cnhuhbert_model = self.cnhuhbert_model.to(self.configs.device)
        
        
        
    def init_bert_weights(self, base_path: str):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
        if self.configs.is_half:
            self.bert_model = self.bert_model.half()
        self.bert_model = self.bert_model.to(self.configs.device)
        
        

    def init_vits_weights(self, weights_path: str):
        self.configs.vits_weights_path = weights_path
        self.configs.save_configs()
        dict_s2 = torch.load(weights_path, map_location=self.configs.device)
        hps = dict_s2["config"]
        self.configs.filter_length = hps["data"]["filter_length"]
        self.configs.segment_size = hps["train"]["segment_size"]
        self.configs.sampling_rate = hps["data"]["sampling_rate"]       
        self.configs.hop_length = hps["data"]["hop_length"]
        self.configs.win_length = hps["data"]["win_length"]
        self.configs.n_speakers = hps["data"]["n_speakers"]
        self.configs.semantic_frame_rate = "25hz"
        kwargs = hps["model"]
        vits_model = SynthesizerTrn(
            self.configs.filter_length // 2 + 1,
            self.configs.segment_size // self.configs.hop_length,
            n_speakers=self.configs.n_speakers,
            **kwargs
        )
        # if ("pretrained" not in weights_path):
        if hasattr(vits_model, "enc_q"):
            del vits_model.enc_q
            
        if self.configs.is_half:
            vits_model = vits_model.half()
        vits_model = vits_model.to(self.configs.device)
        vits_model.eval()
        vits_model.load_state_dict(dict_s2["weight"], strict=False)
        self.vits_model = vits_model

        
    def init_t2s_weights(self, weights_path: str):
        self.configs.t2s_weights_path = weights_path
        self.configs.save_configs()
        self.configs.hz = 50
        dict_s1 = torch.load(weights_path, map_location=self.configs.device)
        config = dict_s1["config"]
        self.configs.max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if self.configs.is_half:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(self.configs.device)
        t2s_model.eval()
        self.t2s_model = t2s_model
        
    def set_ref_audio(self, ref_audio_path:str):
        self._set_prompt_semantic(ref_audio_path)
        self._set_ref_spepc(ref_audio_path)
        
    def _set_ref_spepc(self, ref_audio_path):
        audio = load_audio(ref_audio_path, int(self.configs.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.configs.filter_length,
            self.configs.sampling_rate,
            self.configs.hop_length,
            self.configs.win_length,
            center=False,
        )
        spec = spec.to(self.configs.device)
        if self.configs.is_half:
            spec = spec.half()
        # self.refer_spepc = spec
        self.prompt_cache["refer_spepc"] = spec
        
        
    def _set_prompt_semantic(self, ref_wav_path:str):
        zero_wav = np.zeros(
            int(self.configs.sampling_rate * 0.3),
            dtype=np.float16 if self.configs.is_half else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(self.configs.device)
            zero_wav_torch = zero_wav_torch.to(self.configs.device)
            if self.configs.is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()

            wav16k = torch.cat([wav16k, zero_wav_torch])
            hubert_feature = self.cnhuhbert_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.vits_model.extract_latent(hubert_feature)
    
            prompt_semantic = codes[0, 0].to(self.configs.device)
            self.prompt_cache["prompt_semantic"] = prompt_semantic
    
    def batch_sequences(self, sequences: List[torch.Tensor], axis: int = 0, pad_value: int = 0, max_length:int=None):
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
    
    def to_batch(self, data:list, prompt_data:dict=None, batch_size:int=5, threshold:float=0.75):
        
        _data:list = []
        index_and_len_list = []
        for idx, item in enumerate(data):
            norm_text_len = len(item["norm_text"])
            index_and_len_list.append([idx, norm_text_len])

        index_and_len_list.sort(key=lambda x: x[1])
        # index_and_len_batch_list = [index_and_len_list[idx:min(idx+batch_size,len(index_and_len_list))] for idx in range(0,len(index_and_len_list),batch_size)]
        index_and_len_list = np.array(index_and_len_list, dtype=np.int64)
        
        # for batch_idx, index_and_len_batch in enumerate(index_and_len_batch_list):
        
        batch_index_list = []
        batch_index_list_len = 0
        pos = 0
        while pos <index_and_len_list.shape[0]:
            # batch_index_list.append(index_and_len_list[pos:min(pos+batch_size,len(index_and_len_list))])
            pos_end = min(pos+batch_size,index_and_len_list.shape[0])
            while pos < pos_end:
                batch=index_and_len_list[pos:pos_end, 1].astype(np.float32)
                score=batch[(pos_end-pos)//2]/batch.mean()
                if (score>=threshold) or (pos_end-pos==1):
                    batch_index=index_and_len_list[pos:pos_end, 0].tolist()
                    batch_index_list_len += len(batch_index)
                    batch_index_list.append(batch_index)
                    pos = pos_end
                    break
                pos_end=pos_end-1
        
        assert batch_index_list_len == len(data)
                
        for batch_idx, index_list in enumerate(batch_index_list):
            item_list = [data[idx] for idx in index_list]
            phones_list = []
            # bert_features_list = []
            all_phones_list = []
            all_bert_features_list = []
            norm_text_batch = []
            bert_max_len = 0
            phones_max_len = 0
            for item in item_list:
                if prompt_data is not None:
                    all_bert_features = torch.cat([prompt_data["bert_features"].clone(), item["bert_features"]], 1)
                    all_phones = torch.LongTensor(prompt_data["phones"]+item["phones"])
                    phones = torch.LongTensor(item["phones"])
                    # norm_text = prompt_data["norm_text"]+item["norm_text"]
                else:
                    all_bert_features = item["bert_features"]
                    phones = torch.LongTensor(item["phones"])
                    all_phones = phones.clone()
                    # norm_text = item["norm_text"]
                bert_max_len = max(bert_max_len, all_bert_features.shape[-1])
                phones_max_len = max(phones_max_len, phones.shape[-1])
                
                phones_list.append(phones)
                all_phones_list.append(all_phones)
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])
            # phones_batch = phones_list
            max_len = max(bert_max_len, phones_max_len)
            phones_batch = self.batch_sequences(phones_list, axis=0, pad_value=0, max_length=max_len)
            all_phones_batch = self.batch_sequences(all_phones_list, axis=0, pad_value=0, max_length=max_len)
            all_bert_features_batch = torch.FloatTensor(len(item_list), 1024, max_len)
            all_bert_features_batch.zero_()

            for idx, item in enumerate(all_bert_features_list):
                if item != None:
                    all_bert_features_batch[idx, :, : item.shape[-1]] = item
            
            batch = {
                "phones": phones_batch,
                "all_phones": all_phones_batch,
                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch
            }
            _data.append(batch)
        
        return _data, batch_index_list
    
    def recovery_order(self, data:list, batch_index_list:list)->list:
        lenght = len(sum(batch_index_list, []))
        _data = [None]*lenght
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                _data[index] = data[i][j]
        return _data


        
    
    def run(self, inputs:dict):
        """
        Text to speech inference.
        
        Args:
            inputs (dict): 
                {
                    "text": "",
                    "text_lang: "",
                    "ref_audio_path": "",
                    "prompt_text": "",
                    "prompt_lang": "",
                    "top_k": 5,
                    "top_p": 0.9,
                    "temperature": 0.6,
                    "text_split_method": "",
                    "batch_size": 1,
                    "batch_threshold": 0.75,
                    "speed_factor":1.0,
                }
        returns:
            tulpe[int, np.ndarray]: sampling rate and audio data.
        """
        text:str = inputs.get("text", "")
        text_lang:str = inputs.get("text_lang", "")
        ref_audio_path:str = inputs.get("ref_audio_path", "")
        prompt_text:str = inputs.get("prompt_text", "")
        prompt_lang:str = inputs.get("prompt_lang", "")
        top_k:int = inputs.get("top_k", 20)
        top_p:float = inputs.get("top_p", 0.9)
        temperature:float = inputs.get("temperature", 0.6)
        text_split_method:str = inputs.get("text_split_method", "")
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        
        no_prompt_text = False
        if prompt_text in [None, ""]:
            no_prompt_text = True
        
        assert text_lang in self.configs.langauges
        if not no_prompt_text:
            assert prompt_lang in self.configs.langauges

        if ref_audio_path in [None, ""] and \
            ((self.prompt_cache["prompt_semantic"] is None) or (self.prompt_cache["refer_spepc"] is None)):
            raise ValueError("ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()")

        t0 = ttime()
        if (ref_audio_path is not None) and (ref_audio_path != self.prompt_cache["ref_audio_path"]):
                self.set_ref_audio(ref_audio_path)

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_lang != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
            if self.prompt_cache["prompt_text"] != prompt_text:
                self.prompt_cache["prompt_text"] = prompt_text
                self.prompt_cache["prompt_lang"] = prompt_lang
                phones, bert_features, norm_text = \
                    self.text_preprocessor.segment_and_extract_feature_for_text(
                                                                        prompt_text, 
                                                                        prompt_lang)
                self.prompt_cache["phones"] = phones
                self.prompt_cache["bert_features"] = bert_features
                self.prompt_cache["norm_text"] = norm_text
                
        zero_wav = np.zeros(
            int(self.configs.sampling_rate * 0.3),
            dtype=np.float16 if self.configs.is_half else np.float32,
        )
        
        
        data = self.text_preprocessor.preprocess(text, text_lang, text_split_method)
        audio = []
        t1 = ttime()
        data, batch_index_list = self.to_batch(data, 
                             prompt_data=self.prompt_cache if not no_prompt_text else None, 
                             batch_size=batch_size, 
                             threshold=batch_threshold)
        t2 = ttime()
        zero_wav = torch.zeros(
                        int(self.configs.sampling_rate * 0.3),
                        dtype=torch.float16 if self.configs.is_half else torch.float32,
                        device=self.configs.device
                    )
        
        t_34 = 0.0
        t_45 = 0.0
        for item in data:
            t3 = ttime()
            batch_phones = item["phones"]
            all_phoneme_ids = item["all_phones"]
            all_bert_features = item["all_bert_features"]
            norm_text = item["norm_text"]
            
            # phones = phones.to(self.configs.device)
            all_phoneme_ids = all_phoneme_ids.to(self.configs.device)
            all_bert_features = all_bert_features.to(self.configs.device)
            if self.configs.is_half:
                all_bert_features = all_bert_features.half()
            # all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]*all_phoneme_ids.shape[0], device=self.configs.device)
    
            print(i18n("前端处理后的文本(每句):"), norm_text)
            if no_prompt_text :
                prompt = None
            else:
                prompt = self.prompt_cache["prompt_semantic"].clone().repeat(all_phoneme_ids.shape[0], 1).to(self.configs.device)
            
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    None,
                    prompt,
                    all_bert_features,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.configs.hz * self.configs.max_sec,
                )
            t4 = ttime()
            t_34 += t4 - t3
            
            refer_audio_spepc:torch.Tensor = self.prompt_cache["refer_spepc"].clone().to(self.configs.device)
            if self.configs.is_half:
                refer_audio_spepc = refer_audio_spepc.half()
            
            ## 直接对batch进行decode 生成的音频会有问题
            # pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
            # pred_semantic = self.batch_sequences(pred_semantic_list, axis=0, pad_value=0).unsqueeze(0)
            # batch_phones = batch_phones.to(self.configs.device)
            # batch_audio_fragment =(self.vits_model.decode(
            #         pred_semantic, batch_phones, refer_audio_spepc
            #     ).detach()[:, 0, :])
            # max_audio=torch.abs(batch_audio_fragment).max()#简单防止16bit爆音
            # if max_audio>1: batch_audio_fragment/=max_audio
            # batch_audio_fragment = batch_audio_fragment.cpu().numpy()
            
            ## 改成串行处理
            batch_audio_fragment = []
            for i, idx in enumerate(idx_list):
                phones = batch_phones[i].clone().unsqueeze(0).to(self.configs.device)
                _pred_semantic = (pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0))   # .unsqueeze(0)#mq要多unsqueeze一次
                audio_fragment =(self.vits_model.decode(
                        _pred_semantic, phones, refer_audio_spepc
                    ).detach()[0, 0, :])
                max_audio=torch.abs(audio_fragment).max()#简单防止16bit爆音
                if max_audio>1: audio_fragment/=max_audio
                audio_fragment = torch.cat([audio_fragment, zero_wav], dim=0)
                batch_audio_fragment.append(
                    audio_fragment.cpu().numpy()
                )  ###试试重建不带上prompt部分
            
            audio.append(batch_audio_fragment)
            # audio.append(zero_wav)
            t5 = ttime()
            t_45 += t5 - t4
            
        audio = self.recovery_order(audio, batch_index_list)
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t_34, t_45))
        
        audio = np.concatenate(audio, 0)
        audio = (audio * 32768).astype(np.int16) 
        
        try:
            if speed_factor != 1.0:
                audio = speed_change(audio, speed=speed_factor, sr=int(self.configs.sampling_rate))
        except Exception as e:
            print(f"Failed to change speed of audio: \n{e}")
        
        yield self.configs.sampling_rate, audio
       
       
       
       
def speed_change(input_audio:np.ndarray, speed:float, sr:int):
    # 将 NumPy 数组转换为原始 PCM 流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio