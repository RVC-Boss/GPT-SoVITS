import torch
import torch.nn.functional as F
import torchaudio
from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule
from feature_extractor import cnhubert
from module.models_onnx import SynthesizerTrn, symbols_v1, symbols_v2
from torch import nn
from sv import SV
import onnx
from onnx import helper, TensorProto
cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
from transformers import HubertModel, HubertConfig
import os
import json
from text import cleaned_text_to_sequence
import onnxsim
from onnxconverter_common import float16

def simplify_onnx_model(onnx_model_path: str):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    # Simplify the model
    model_simplified, _ = onnxsim.simplify(model)
    # Save the simplified model
    onnx.save(model_simplified, onnx_model_path)

def convert_onnx_to_half(onnx_model_path:str):
    try:
        model = onnx.load(onnx_model_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, onnx_model_path)
    except Exception as e:
        print(f"Error converting {onnx_model_path} to half precision: {e}")


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample audio from orig_sr to target_sr using linear interpolation.
    audio: (batch, channels, samples) or (channels, samples) or (samples,)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    # audio shape: (batch, channels, samples)
    batch, channels, samples = audio.shape
    # Reshape to combine batch and channels for interpolation
    audio = audio.reshape(batch * channels, 1, samples)
    # Use scale_factor instead of a computed size for ONNX export compatibility
    resampled = F.interpolate(audio, scale_factor=target_sr / orig_sr, mode='linear', align_corners=False)
    new_samples = resampled.shape[-1]
    resampled = resampled.reshape(batch, channels, new_samples)
    resampled = resampled.squeeze(0).squeeze(0)
    return resampled


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


class T2SInitStage(nn.Module):
    def __init__(self, t2s, vits):
        super().__init__()
        self.encoder = t2s.onnx_encoder
        self.vits = vits
        self.num_layers = t2s.num_layers

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        codes = self.vits.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)
        prompt = prompt_semantic.unsqueeze(0)
        x = self.encoder(all_phoneme_ids, bert)

        x_seq_len = torch.onnx.operators.shape_as_tensor(x)[1]
        y_seq_len = torch.onnx.operators.shape_as_tensor(prompt)[1]

        init_k = torch.zeros((self.num_layers, (x_seq_len + y_seq_len), 1, 512), dtype=torch.float)
        init_v = torch.zeros((self.num_layers, (x_seq_len + y_seq_len), 1, 512), dtype=torch.float)

        return x, prompt, init_k, init_v, x_seq_len, y_seq_len

class T2SModel(nn.Module):
    def __init__(self, t2s_path, vits_model):
        super().__init__()
        dict_s1 = torch.load(t2s_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "ojbk", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        self.t2s_model.eval()
        self.vits_model = vits_model.vq_model
        self.hz = 50
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model.model.top_k = torch.LongTensor([self.config["inference"]["top_k"]])
        self.t2s_model.model.early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        self.t2s_model = self.t2s_model.model
        self.t2s_model.init_onnx()
        self.init_stage = T2SInitStage(self.t2s_model, self.vits_model)
        self.stage_decoder = self.t2s_model.stage_decoder

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, top_k=None, top_p=None, repetition_penalty=None, temperature=None):
        x, prompt, init_k, init_v, x_seq_len, y_seq_len = self.init_stage(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        empty_tensor = torch.empty((1,0,512)).to(torch.float)
        # first step
        y, k, v, y_emb, logits, samples = self.stage_decoder(x, prompt, init_k, init_v, 
                          empty_tensor, 
                          top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature, 
                          first_infer=torch.LongTensor([1]), x_seq_len=x_seq_len, y_seq_len=y_seq_len)

        for idx in range(5): # This is a fake one! DO NOT take this as reference
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 1))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 1))
            y_seq_len = y.shape[1]
            y, k, v, y_emb, logits, samples = self.stage_decoder(empty_tensor, y, k, v, 
                                                                 y_emb, 
                                                                 top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature, 
                                                                 first_infer=torch.LongTensor([0]), x_seq_len=x_seq_len, y_seq_len=y_seq_len)
            # if torch.argmax(logits, dim=-1)[0] == self.t2s_model.EOS or samples[0, 0] == self.t2s_model.EOS:
            #     break

        return y[:, -5:].unsqueeze(0)

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, top_k=None, top_p=None, repetition_penalty=None, temperature=None):
        torch.onnx.export(
            self.init_stage,
            (ref_seq, text_seq, ref_bert, text_bert, ssl_content),
            f"onnx/{project_name}/{project_name}_t2s_init_stage.onnx",
            input_names=["ref_text_phones", "input_text_phones", "ref_text_bert", "input_text_bert", "hubert_ssl_content"],
            output_names=["x", "prompt", "init_k", "init_v", 'x_seq_len', 'y_seq_len'],
            dynamic_axes={
                "ref_text_phones": {1: "ref_length"},
                "input_text_phones": {1: "text_length"},
                "ref_text_bert": {0: "ref_length"},
                "input_text_bert": {0: "text_length"},
                "hubert_ssl_content": {2: "ssl_length"},
            },
            opset_version=16,
            do_constant_folding=False
        )
        simplify_onnx_model(f"onnx/{project_name}/{project_name}_t2s_init_stage.onnx")
        x, prompt, init_k, init_v, x_seq_len, y_seq_len = self.init_stage(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        empty_tensor = torch.empty((1,0,512)).to(torch.float)
        x_seq_len = torch.Tensor([x_seq_len]).to(torch.int64)
        y_seq_len = torch.Tensor([y_seq_len]).to(torch.int64)

        y, k, v, y_emb, logits, samples = self.stage_decoder(x, prompt, init_k, init_v, 
                                                             empty_tensor, 
                                                             top_k, top_p, repetition_penalty, temperature, 
                                                             torch.LongTensor([1]), x_seq_len, y_seq_len)
        print(y.shape, k.shape, v.shape, y_emb.shape, logits.shape, samples.shape)
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 1))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 1))
        y_seq_len = torch.Tensor([y.shape[1]]).to(torch.int64)

        torch.onnx.export(
            self.stage_decoder,
            (x, y, k, v, y_emb, top_k, top_p, repetition_penalty, temperature, torch.LongTensor([0]), x_seq_len, y_seq_len),
            f"onnx/{project_name}/{project_name}_t2s_stage_decoder.onnx",
            input_names=["ix", "iy", "ik", "iv", "iy_emb", "top_k", "top_p", "repetition_penalty", "temperature", "if_init_step", "x_seq_len", "y_seq_len"],
            output_names=["y", "k", "v", "y_emb", "logits", "samples"],
            dynamic_axes={
                "ix": {1: "ix_length"},
                "iy": {1: "iy_length"},
                "ik": {1: "ik_length"},
                "iv": {1: "iv_length"},
                "iy_emb": {1: "iy_emb_length"},
            },
            verbose=False,
            opset_version=16,
        )
        simplify_onnx_model(f"onnx/{project_name}/{project_name}_t2s_stage_decoder.onnx")


class VitsModel(nn.Module):
    def __init__(self, vits_path, version:str = 'v2'):
        super().__init__()
        dict_s2 = torch.load(vits_path, map_location="cpu", weights_only=False)
        self.hps = dict_s2["config"]
        if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            self.hps["model"]["version"] = "v1"
        else:
            self.hps["model"]["version"] = version

        self.is_v2p = version.lower() in ['v2pro', 'v2proplus']

        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        # print(f"filter_length:{self.hps.data.filter_length} sampling_rate:{self.hps.data.sampling_rate} hop_length:{self.hps.data.hop_length} win_length:{self.hps.data.win_length}")
        #v2 filter_length: 2048 sampling_rate: 32000 hop_length: 640 win_length: 2048
    def forward(self, text_seq, pred_semantic, spectrum, sv_emb):
        if self.is_v2p:
            return self.vq_model(pred_semantic, text_seq, spectrum, sv_emb=sv_emb)[0, 0]
        else:
            return self.vq_model(pred_semantic, text_seq, spectrum)[0, 0]


class GptSoVits(nn.Module):
    def __init__(self, vits, t2s):
        super().__init__()
        self.vits = vits
        self.t2s = t2s

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, spectrum, sv_emb, top_k=None, top_p=None, repetition_penalty=None, temperature=None):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
        audio = self.vits(text_seq, pred_semantic, spectrum, sv_emb)
        return audio

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, spectrum, sv_emb, project_name, top_k=None, top_p=None, repetition_penalty=None, temperature=None):
        self.t2s.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
        torch.onnx.export(
            self.vits,
            (text_seq, pred_semantic, spectrum, sv_emb),
            f"onnx/{project_name}/{project_name}_vits.onnx",
            input_names=["input_text_phones", "pred_semantic", "spectrum", "sv_emb"],
            output_names=["audio"],
            dynamic_axes={
                "input_text_phones": {1: "text_length"},
                "pred_semantic": {2: "pred_length"},
                "spectrum": {2: "spectrum_length"},
            },
            opset_version=17,
            verbose=False,
        )
        simplify_onnx_model(f"onnx/{project_name}/{project_name}_vits.onnx")


class AudioPreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the model
        self.model = HubertModel.from_pretrained(cnhubert_base_path, local_files_only=True)
        self.model.eval()

        self.sv_model = SV("cpu", False)

    def forward(self, ref_audio_32k):
        spectrum = spectrogram_torch(
            ref_audio_32k,
            2048,
            32000,
            640,
            2048,
            center=False,
        )
        ref_audio_16k = resample_audio(ref_audio_32k, 32000, 16000)

        sv_emb = self.sv_model.compute_embedding3_onnx(ref_audio_16k)

        zero_tensor = torch.zeros((1, 9600), dtype=torch.float32)
        ref_audio_16k = ref_audio_16k.unsqueeze(0)
        # concate zero_tensor with waveform
        ref_audio_16k = torch.cat([ref_audio_16k, zero_tensor], dim=1)
        ssl_content = self.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)

        return ssl_content, spectrum, sv_emb

def export(vits_path, gpt_path, project_name, voice_model_version, export_audio_preprocessor=True, half_precision=False):
    vits = VitsModel(vits_path, version=voice_model_version)
    gpt = T2SModel(gpt_path, vits)
    gpt_sovits = GptSoVits(vits, gpt)
    preprocessor = AudioPreprocess()
    ref_seq = torch.LongTensor(
        [
            cleaned_text_to_sequence(
                [
                    "n",
                    "i2",
                    "h",
                    "ao3",
                    ",",
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                ],
                version='v2',
            )
        ]
    )
    text_seq = torch.LongTensor(
        [
            cleaned_text_to_sequence(
                [
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                ],
                version='v2',
            )
        ]
    )
    ref_bert = torch.randn((ref_seq.shape[1], 1024)).float()
    text_bert = torch.randn((text_seq.shape[1], 1024)).float()
    ref_audio32k = torch.randn((1, 32000 * 5)).float() - 0.5 # 5 seconds of dummy audio
    top_k = torch.LongTensor([15])
    top_p = torch.FloatTensor([1.0])
    repetition_penalty = torch.FloatTensor([1.0])
    temperature = torch.FloatTensor([1.0])

    os.makedirs(f"onnx/{project_name}", exist_ok=True)

    [ssl_content, spectrum, sv_emb] = preprocessor(ref_audio32k)
    gpt_sovits(ref_seq, text_seq, ref_bert, text_bert, ssl_content.float(), spectrum.float(), sv_emb.float(), top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
    # exit()
    gpt_sovits.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content.float(), spectrum.float(), sv_emb.float(), project_name, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)

    if export_audio_preprocessor:
        torch.onnx.export(preprocessor, (ref_audio32k,), f"onnx/{project_name}/{project_name}_audio_preprocess.onnx",
                    input_names=["audio32k"],
                    output_names=["hubert_ssl_output", "spectrum", "sv_emb"],
                    dynamic_axes={
                        "audio32k": {1: "sequence_length"},
                        "hubert_ssl_output": {2: "hubert_length"},
                        "spectrum": {2: "spectrum_length"}
                    })
        simplify_onnx_model(f"onnx/{project_name}/{project_name}_audio_preprocess.onnx")
        
    if half_precision:
        if export_audio_preprocessor:
            convert_onnx_to_half(f"onnx/{project_name}/{project_name}_audio_preprocess.onnx")
        convert_onnx_to_half(f"onnx/{project_name}/{project_name}_vits.onnx")
        convert_onnx_to_half(f"onnx/{project_name}/{project_name}_t2s_init_step.onnx")
        convert_onnx_to_half(f"onnx/{project_name}/{project_name}_t2s_stage_step.onnx")

    configJson = {
        "project_name": project_name,
        "type": "GPTSoVITS",
        "version" : voice_model_version,
        "bert_base_path": 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
        "cnhuhbert_base_path": 'GPT_SoVITS/pretrained_models/chinese-hubert-base',
        "t2s_weights_path": gpt_path,
        "vits_weights_path": vits_path,
        "half_precision": half_precision
    }
    with open(f"onnx/{project_name}/config.json", "w", encoding="utf-8") as f:
        json.dump(configJson, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    try:
        os.mkdir("onnx")
    except:
        pass

    # 因为io太频繁，可能导致模型导出出错(wsl非常明显)，请自行重试

    gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    vits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
    exp_path = "v1_export"
    version = "v1"
    export(vits_path, gpt_path, exp_path, version)

    gpt_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    vits_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    exp_path = "v2_export"
    version = "v2"
    export(vits_path, gpt_path, exp_path, version)
    

    gpt_path = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    vits_path = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
    exp_path = "v2pro_export"
    version = "v2Pro"
    export(vits_path, gpt_path, exp_path, version)

    gpt_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    vits_path = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
    exp_path = "v2proplus_export"
    version = "v2ProPlus"
    export(vits_path, gpt_path, exp_path, version)


