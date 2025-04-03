import os
import json
import onnx
import torch
import onnxsim

from torch.nn import Module
from feature_extractor import cnhubert
from onnxruntime import InferenceSession
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, AutoModelForMaskedLM
import AR.models.t2s_model_onnx as t2s

from module.models_onnx import SynthesizerTrn

root_path = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(root_path, "onnx")
if not os.path.exists(onnx_path):
    os.makedirs(onnx_path)

class BertWrapper(Module):
    def __init__(self):
        bert_path = os.environ.get(
            "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        )
        super(BertWrapper, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)

    def forward(self, input_ids):
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        res = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        return torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    
    def export_onnx(self):
        vocab_dict = { k: v for k, v in self.tokenizer.get_vocab().items() }
        vocab_path = os.path.join(onnx_path, "Vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(vocab_dict, f, indent=4)
        dummy_input = torch.randint(0, 100, (1, 20)).long()
        torch.onnx.export(
            self,
            dummy_input,
            os.path.join(onnx_path, "Bert.onnx"),
            input_names=["input_ids"],
            output_names=["output"],
            dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
            opset_version=18,
        )
        sim, _ = onnxsim.simplify(os.path.join(onnx_path, "Bert.onnx"))
        onnx.save_model(sim, os.path.join(onnx_path, "Bert.onnx"))
        print("Exported BERT to ONNX format.")


class CnHubertWrapper(Module):
    def __init__(self):
        super(CnHubertWrapper, self).__init__()
        cnhubert_base_path = os.environ.get(
            "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        )
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.model = cnhubert.get_model().model

    def forward(self, signal):
        return self.model(signal)["last_hidden_state"]
    
    def export_onnx(self):
        dummy_input = torch.randn(1, 16000 * 10)
        torch.onnx.export(
            self,
            dummy_input,
            os.path.join(onnx_path, "CnHubert.onnx"),
            input_names=["signal"],
            output_names=["output"],
            dynamic_axes={"signal": {0: "batch_size", 1: "sequence_length"}},
            opset_version=18,
        )
        sim, _ = onnxsim.simplify(os.path.join(onnx_path, "CnHubert.onnx"))
        onnx.save_model(sim, os.path.join(onnx_path, "CnHubert.onnx"))
        print("Exported CN-Hubert to ONNX format.")


class Text2SemanticLightningModule(LightningModule):
    def __init__(self, path, top_k=20, cache_size=2000):
        super().__init__()
        dict_s1 = torch.load(path, map_location="cpu")
        config = dict_s1["config"]
        self.model = t2s.Text2SemanticDecoder(config=config)
        self.load_state_dict(dict_s1["weight"])
        self.cache_size = cache_size
        self.top_k = top_k

def export_ar(path, top_k=20, cache_size=2000):
    model_l = Text2SemanticLightningModule(path, top_k=top_k, cache_size=cache_size)
    model = model_l.model

    x = torch.randint(0, 100, (1, 20)).long()
    x_len = torch.tensor([20]).long()
    y = torch.randint(0, 100, (1, 20)).long()
    y_len = torch.tensor([20]).long()
    bert_feature = torch.randn(1, 20, 1024)
    top_p = torch.tensor([0.8])
    repetition_penalty = torch.tensor([1.35])
    temperature = torch.tensor([0.6])

    prompt_processor = t2s.PromptProcessor(cache_len=cache_size, model=model, top_k=top_k)
    decode_next_token = t2s.DecodeNextToken(cache_len=cache_size, model=model, top_k=top_k)

    torch.onnx.export(
        prompt_processor,
        (x, x_len, y, y_len, bert_feature, top_p, repetition_penalty, temperature),
        os.path.join(onnx_path, "PromptProcessor.onnx"),
        input_names=["x", "x_len", "y", "y_len", "bert_feature", "top_p", "repetition_penalty", "temperature"],
        output_names=["y", "k_cache", "v_cache", "xy_pos", "y_idx", "samples"],
        dynamic_axes={
            "x": {0: "batch_size", 1: "sequence_length"},
            "y": {0: "batch_size", 1: "sequence_length"},
            "bert_feature": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=18,
    )

    sim, _ = onnxsim.simplify(os.path.join(onnx_path, "PromptProcessor.onnx"))
    onnx.save_model(sim, os.path.join(onnx_path, "PromptProcessor.onnx"))

    y, k_cache, v_cache, xy_pos, y_idx, samples = prompt_processor(
        x, x_len, y, y_len, bert_feature, top_p, repetition_penalty, temperature
    )

    torch.onnx.export(
        decode_next_token,
        (y, k_cache, v_cache, xy_pos, y_idx, top_p, repetition_penalty, temperature),
        os.path.join(onnx_path, "DecodeNextToken.onnx"),
        input_names=["y", "k_cache", "v_cache", "xy_pos", "y_idx", "top_p", "repetition_penalty", "temperature"],
        output_names=["y", "k_cache", "v_cache", "xy_pos", "y_idx", "samples"],
        dynamic_axes={
            "y": {0: "batch_size", 1: "sequence_length"},
            "k_cache": {1: "batch_size", 2: "sequence_length"},
            "v_cache": {1: "batch_size", 2: "sequence_length"},
        },
        opset_version=18
    )

    sim, _ = onnxsim.simplify(os.path.join(onnx_path, "DecodeNextToken.onnx"))
    onnx.save_model(sim, os.path.join(onnx_path, "DecodeNextToken.onnx"))


from io import BytesIO
def load_sovits_new(sovits_path):
    f=open(sovits_path,"rb")
    meta=f.read(2)
    if meta!="PK":
        data = b'PK' + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path,map_location="cpu", weights_only=False)


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


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )
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


class Extractor(Module):
    def __init__(self, model):
        super(Extractor, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.extract_latent(x.transpose(1, 2))


class V1V2(Module):
    def __init__(self, path):
        super(V1V2, self).__init__()
        dict_s2 = load_sovits_new(path)
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if 'enc_p.text_embedding.weight'not in dict_s2['weight']:
            hps.model.version = "v2"#v3model,v2sybomls
        elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"
        version=hps.model.version
        # print("sovits版本:",hps.model.version)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.eval()
        self.vq_model = vq_model
        self.hps = hps
        self.ext = Extractor(self.vq_model)

    def forward(self, text_seq, pred_semantic, ref_audio):
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False
        )
        return self.vq_model(pred_semantic.unsqueeze(0), text_seq, refer)[0, 0]
    
    def export(self):
        test_seq = torch.randint(0, 100, (1, 20)).long()
        pred_semantic = torch.randint(0, 100, (1, 20)).long()
        ref_audio = torch.randn(1, 16000 * 10)
        torch.onnx.export(
            self,
            (test_seq, pred_semantic, ref_audio),
            os.path.join(onnx_path, "GptSoVitsV1V2.onnx"),
            input_names=["text_seq", "pred_semantic", "ref_audio"],
            output_names=["output"],
            dynamic_axes={
                "text_seq": {0: "batch_size", 1: "sequence_length"},
                "pred_semantic": {0: "batch_size", 1: "sequence_length"},
                "ref_audio": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=18,
        )

        sim, _ = onnxsim.simplify(os.path.join(onnx_path, "GptSoVitsV1V2.onnx"))
        onnx.save_model(sim, os.path.join(onnx_path, "GptSoVitsV1V2.onnx"))
        ref_units = torch.randn(1, 20, 768)
        torch.onnx.export(
            self.ext,
            ref_units,
            os.path.join(onnx_path, "Extractor.onnx"),
            input_names=["ref_units"],
            output_names=["output"],
            dynamic_axes={
                "ref_units": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=18,
        )

        
if __name__ == "__main__":
    #CnHubertWrapper().export_onnx()
    #BertWrapper().export_onnx()
    V1V2("D:\\VSGIT\GPT-SoVITS-main\\GPT_SoVITS\\GPT-SoVITS-v3lora-20250228\\GPT_SoVITS\\t\\SoVITS_weights\\小特.pth").export()
    '''export_ar(
        "D:\\VSGIT\GPT-SoVITS-main\\GPT_SoVITS\\GPT-SoVITS-v3lora-20250228\\GPT_SoVITS\\t\\GPT_weights\\小特.ckpt",
        top_k=10,
        cache_size=1500,
    )'''
    
