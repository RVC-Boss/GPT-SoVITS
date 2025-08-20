import torch
import torch.nn.functional as F
import torchaudio
from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule
from feature_extractor import cnhubert
from module.models_onnx import SynthesizerTrn, symbols_v1, symbols_v2
from torch import nn
from sv import SV

cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
from transformers import HubertModel, HubertConfig
import json
import os

import soundfile
from tqdm import tqdm
from text import cleaned_text_to_sequence


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


class T2SEncoder(nn.Module):
    def __init__(self, t2s, vits):
        super().__init__()
        self.encoder = t2s.onnx_encoder
        self.fsdc = t2s.first_stage_decoder
        self.vits = vits

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        codes = self.vits.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)
        prompt = prompt_semantic.unsqueeze(0)
        return self.fsdc(self.encoder(all_phoneme_ids, bert), prompt)
        


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
        self.onnx_encoder = T2SEncoder(self.t2s_model, self.vits_model)
        self.first_stage_decoder = self.t2s_model.first_stage_decoder
        self.stage_decoder = self.t2s_model.stage_decoder
        # self.t2s_model = torch.jit.script(self.t2s_model)

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.t2s_model.early_stop_num

        # [1,N] [1,N] [N, 1024] [N, 1024] [1, 768, N]
        y, k, v, y_emb, x_example = self.onnx_encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)

        for idx in tqdm(range(1, 20)): # This is a fake one! do take this as reference
            # [1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
            enco = self.stage_decoder(y, k, v, y_emb, x_example)
            y, k, v, y_emb, logits, samples = enco
            if torch.argmax(logits, dim=-1)[0] == self.t2s_model.EOS or samples[0, 0] == self.t2s_model.EOS:
                break
        y[0, -1] = 0

        return y[:, -idx:].unsqueeze(0)

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, dynamo=False):
        # self.onnx_encoder = torch.jit.script(self.onnx_encoder)
        if dynamo:
            export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
            onnx_encoder_export_output = torch.onnx.dynamo_export(
                self.onnx_encoder, (ref_seq, text_seq, ref_bert, text_bert, ssl_content), export_options=export_options
            )
            onnx_encoder_export_output.save(f"onnx/{project_name}/{project_name}_t2s_encoder.onnx")
            return

        torch.onnx.export(
            self.onnx_encoder,
            (ref_seq, text_seq, ref_bert, text_bert, ssl_content),
            f"onnx/{project_name}/{project_name}_t2s_encoder.onnx",
            input_names=["ref_seq", "text_seq", "ref_bert", "text_bert", "ssl_content"],
            output_names=["y", "k", "v", "y_emb", "x_example"],
            dynamic_axes={
                "ref_seq": {1: "ref_length"},
                "text_seq": {1: "text_length"},
                "ref_bert": {0: "ref_length"},
                "text_bert": {0: "text_length"},
                "ssl_content": {2: "ssl_length"},
            },
            opset_version=16,
        )
        y, k, v, y_emb, x_example = self.onnx_encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)

        # torch.onnx.export(
        #     self.first_stage_decoder,
        #     (x, prompts),
        #     f"onnx/{project_name}/{project_name}_t2s_fsdec.onnx",
        #     input_names=["x", "prompts"],
        #     output_names=["y", "k", "v", "y_emb", "x_example"],
        #     dynamic_axes={
        #         "x": {1: "x_length"},
        #         "prompts": {1: "prompts_length"},
        #     },
        #     verbose=False,
        #     opset_version=16,
        # )
        # y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)

        torch.onnx.export(
            self.stage_decoder,
            (y, k, v, y_emb, x_example),
            f"onnx/{project_name}/{project_name}_t2s_sdec.onnx",
            input_names=["iy", "ik", "iv", "iy_emb", "ix_example"],
            output_names=["y", "k", "v", "y_emb", "logits", "samples"],
            dynamic_axes={
                "iy": {1: "iy_length"},
                "ik": {1: "ik_length"},
                "iv": {1: "iv_length"},
                "iy_emb": {1: "iy_emb_length"},
                "ix_example": {1: "ix_example_length"},
            },
            verbose=False,
            opset_version=16,
        )


class VitsModel(nn.Module):
    def __init__(self, vits_path, version:str = 'v2'):
        super().__init__()
        dict_s2 = torch.load(vits_path, map_location="cpu")
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
        #filter_length: 2048 sampling_rate: 32000 hop_length: 640 win_length: 2048
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

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, spectrum, sv_emb):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        audio = self.vits(text_seq, pred_semantic, spectrum, sv_emb)
        return audio

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, spectrum, sv_emb, project_name):
        self.t2s.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name)
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        torch.onnx.export(
            self.vits,
            (text_seq, pred_semantic, spectrum, sv_emb),
            f"onnx/{project_name}/{project_name}_vits.onnx",
            input_names=["text_seq", "pred_semantic", "spectrum", "sv_emb"],
            output_names=["audio"],
            dynamic_axes={
                "text_seq": {1: "text_length"},
                "pred_semantic": {2: "pred_length"},
                "ref_audio": {1: "audio_length"},
                "spectrum": {2: "spectrum_length"},
            },
            opset_version=17,
            verbose=False,
        )


class AudioPreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = HubertConfig.from_pretrained(cnhubert_base_path)
        self.config._attn_implementation = "eager"  # Use standard attention
        self.config.apply_spec_augment = False      # Disable masking for inference
        self.config.layerdrop = 0.0                 # Disable layer dropout
        
        # Load the model
        self.model = HubertModel.from_pretrained(
            cnhubert_base_path,
            config=self.config, 
            local_files_only=True
        )
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

        zero_tensor = torch.zeros((1, 4800), dtype=torch.float32)
        ref_audio_16k = ref_audio_16k.unsqueeze(0)
        # concate zero_tensor with waveform
        ref_audio_16k = torch.cat([ref_audio_16k, zero_tensor], dim=1)
        ssl_content = self.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)

        return ssl_content, spectrum, sv_emb


def export(vits_path, gpt_path, project_name, voice_model_version="v2"):
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
    ref_audio = torch.randn((1, 48000 * 5)).float()
    # ref_audio = torch.tensor([load_audio("rec.wav", 48000)]).float()
    ref_audio32k = torchaudio.functional.resample(ref_audio, 48000, 32000).float()

    try:
        os.mkdir(f"onnx/{project_name}")
    except:
        pass

    torch.onnx.export(preprocessor, (ref_audio32k,), f"onnx/{project_name}/{project_name}_audio_preprocess.onnx",
                      input_names=["audio32k"],
                      output_names=["hubert_ssl_output", "spectrum", "sv_emb"],
                      dynamic_axes={
                          "audio32k": {1: "sequence_length"},
                          "hubert_ssl_output": {2: "hubert_length"},
                          "spectrum": {2: "spectrum_length"}
                      })

    [ssl_content, spectrum, sv_emb] = preprocessor(ref_audio32k)
    gpt_sovits(ref_seq, text_seq, ref_bert, text_bert, ssl_content.float(), spectrum.float(), sv_emb.float())
    # exit()
    gpt_sovits.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content.float(), spectrum.float(), sv_emb.float(), project_name)

    if voice_model_version == "v1":
        symbols = symbols_v1
    else:
        symbols = symbols_v2

if __name__ == "__main__":
    try:
        os.mkdir("onnx")
    except:
        pass

    gpt_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    vits_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    exp_path = "v2_export"
    version = "v2"
    export(vits_path, gpt_path, exp_path, version)

    gpt_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    vits_path = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
    exp_path = "v2pro_export"
    version = "v2Pro"
    export(vits_path, gpt_path, exp_path, version)

    gpt_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    vits_path = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
    exp_path = "v2proplus_export"
    version = "v2ProPlus"
    export(vits_path, gpt_path, exp_path, version)


