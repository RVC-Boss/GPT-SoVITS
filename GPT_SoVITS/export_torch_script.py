# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import argparse
from typing import Optional
from my_utils import load_audio
import torch
import torchaudio

from torch import IntTensor, LongTensor, Tensor, nn
from torch.nn import functional as F

from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.models_onnx import SynthesizerTrn

from inference_webui import get_phones_and_bert

import os
import soundfile

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}


def get_raw_t2s_model(dict_s1) -> Text2SemanticLightningModule:
    config = dict_s1["config"]
    config["model"]["dropout"] = float(config["model"]["dropout"])
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_model = t2s_model.eval()
    return t2s_model


@torch.jit.script
def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    # if previous_tokens is not None:
    #     previous_tokens = previous_tokens.squeeze()
    # print(logits.shape,previous_tokens.shape)
    # pdb.set_trace()
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(dim=1, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[:, 0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


@torch.jit.script
def multinomial_sample_one_no_sync(probs_sort):
    # Does multinomial sampling without a cuda synchronization
    q = torch.randn_like(probs_sort)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


@torch.jit.script
def sample(
    logits,
    previous_tokens,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    probs = logits_to_probs(
        logits=logits,
        previous_tokens=previous_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.jit.script
def spectrogram_torch(y: Tensor, n_fft: int, sampling_rate: int, hop_size: int, win_size: int, center: bool = False):
    hann_window = torch.hann_window(win_size, device=y.device, dtype=y.dtype)
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


@torch.jit.script
class T2SMLP:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        x = F.relu(F.linear(x, self.w1, self.b1))
        x = F.linear(x, self.w2, self.b2)
        return x


@torch.jit.script
class T2SBlock:
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp: T2SMLP,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        norm_w1,
        norm_b1,
        norm_eps1: float,
        norm_w2,
        norm_b2,
        norm_eps2: float,
    ):
        self.num_heads = num_heads
        self.mlp = mlp
        self.hidden_dim: int = hidden_dim
        self.qkv_w = qkv_w
        self.qkv_b = qkv_b
        self.out_w = out_w
        self.out_b = out_b
        self.norm_w1 = norm_w1
        self.norm_b1 = norm_b1
        self.norm_eps1 = norm_eps1
        self.norm_w2 = norm_w2
        self.norm_b2 = norm_b2
        self.norm_eps2 = norm_eps2

        self.false = torch.tensor(False, dtype=torch.bool)

    @torch.jit.ignore
    def to_mask(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        if padding_mask is None:
            return x

        if padding_mask.dtype == torch.bool:
            return x.masked_fill(padding_mask, 0)
        else:
            return x * padding_mask

    def process_prompt(self, x: torch.Tensor, attn_mask: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        q, k, v = F.linear(self.to_mask(x, padding_mask), self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]

        q = self.to_mask(q, padding_mask)
        k_cache = self.to_mask(k, padding_mask)
        v_cache = self.to_mask(v, padding_mask)

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)

        attn = attn.permute(2, 0, 1, 3).reshape(batch_size * q_len, self.hidden_dim)
        attn = attn.view(q_len, batch_size, self.hidden_dim).transpose(1, 0)
        attn = F.linear(self.to_mask(attn, padding_mask), self.out_w, self.out_b)

        if padding_mask is not None:
            for i in range(batch_size):
                # mask = padding_mask[i,:,0]
                if self.false.device != padding_mask.device:
                    self.false = self.false.to(padding_mask.device)
                idx = torch.where(padding_mask[i, :, 0] == self.false)[0]
                x_item = x[i, idx, :].unsqueeze(0)
                attn_item = attn[i, idx, :].unsqueeze(0)
                x_item = x_item + attn_item
                x_item = F.layer_norm(x_item, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
                x_item = x_item + self.mlp.forward(x_item)
                x_item = F.layer_norm(
                    x_item,
                    [self.hidden_dim],
                    self.norm_w2,
                    self.norm_b2,
                    self.norm_eps2,
                )
                x[i, idx, :] = x_item.squeeze(0)
            x = self.to_mask(x, padding_mask)
        else:
            x = x + attn
            x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
            x = x + self.mlp.forward(x)
            x = F.layer_norm(
                x,
                [self.hidden_dim],
                self.norm_w2,
                self.norm_b2,
                self.norm_eps2,
            )
        return x, k_cache, v_cache

    def decode_next_token(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        k_cache = torch.cat([k_cache, k], dim=1)
        v_cache = torch.cat([v_cache, v], dim=1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k_cache.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.permute(2, 0, 1, 3).reshape(batch_size * q_len, self.hidden_dim)
        attn = attn.view(q_len, batch_size, self.hidden_dim).transpose(1, 0)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache


@torch.jit.script
class T2STransformer:
    def __init__(self, num_blocks: int, blocks: list[T2SBlock]):
        self.num_blocks: int = num_blocks
        self.blocks = blocks

    def process_prompt(self, x: torch.Tensor, attn_mask: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        k_cache: list[torch.Tensor] = []
        v_cache: list[torch.Tensor] = []
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(x, attn_mask, padding_mask)
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache

    def decode_next_token(self, x: torch.Tensor, k_cache: list[torch.Tensor], v_cache: list[torch.Tensor]):
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token(x, k_cache[i], v_cache[i])
        return x, k_cache, v_cache


class VitsModel(nn.Module):
    def __init__(self, vits_path):
        super().__init__()
        # dict_s2 = torch.load(vits_path,map_location="cpu")
        dict_s2 = torch.load(vits_path)
        self.hps = dict_s2["config"]
        if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            self.hps["model"]["version"] = "v1"
        else:
            self.hps["model"]["version"] = "v2"

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

    def forward(self, text_seq, pred_semantic, ref_audio, speed=1.0):
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return self.vq_model(pred_semantic, text_seq, refer, speed)[0, 0]


class T2SModel(nn.Module):
    def __init__(self, raw_t2s: Text2SemanticLightningModule):
        super(T2SModel, self).__init__()
        self.model_dim = raw_t2s.model.model_dim
        self.embedding_dim = raw_t2s.model.embedding_dim
        self.num_head = raw_t2s.model.num_head
        self.num_layers = raw_t2s.model.num_layers
        self.vocab_size = raw_t2s.model.vocab_size
        self.phoneme_vocab_size = raw_t2s.model.phoneme_vocab_size
        # self.p_dropout = float(raw_t2s.model.p_dropout)
        self.EOS: int = int(raw_t2s.model.EOS)
        self.norm_first = raw_t2s.model.norm_first
        assert self.EOS == self.vocab_size - 1
        self.hz = 50

        self.bert_proj = raw_t2s.model.bert_proj
        self.ar_text_embedding = raw_t2s.model.ar_text_embedding
        self.ar_text_position = raw_t2s.model.ar_text_position
        self.ar_audio_embedding = raw_t2s.model.ar_audio_embedding
        self.ar_audio_position = raw_t2s.model.ar_audio_position

        # self.t2s_transformer = T2STransformer(self.num_layers, blocks)
        # self.t2s_transformer = raw_t2s.model.t2s_transformer

        blocks = []
        h = raw_t2s.model.h

        for i in range(self.num_layers):
            layer = h.layers[i]
            t2smlp = T2SMLP(layer.linear1.weight, layer.linear1.bias, layer.linear2.weight, layer.linear2.bias)

            block = T2SBlock(
                self.num_head,
                self.model_dim,
                t2smlp,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )

            blocks.append(block)

        self.t2s_transformer = T2STransformer(self.num_layers, blocks)

        # self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.ar_predict_layer = raw_t2s.model.ar_predict_layer
        # self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
        self.max_sec = raw_t2s.config["data"]["max_sec"]
        self.top_k = int(raw_t2s.config["inference"]["top_k"])
        self.early_stop_num = torch.LongTensor([self.hz * self.max_sec])

    def forward(
        self,
        prompts: LongTensor,
        ref_seq: LongTensor,
        text_seq: LongTensor,
        ref_bert: torch.Tensor,
        text_bert: torch.Tensor,
        top_k: LongTensor,
    ):
        bert = torch.cat([ref_bert.T, text_bert.T], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)

        x = self.ar_text_embedding(all_phoneme_ids)
        x = x + self.bert_proj(bert.transpose(1, 2))
        x: torch.Tensor = self.ar_text_position(x)

        early_stop_num = self.early_stop_num

        # [1,N,512] [1,N]
        # y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)
        y = prompts
        # x_example = x[:,:,0] * 0.0

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        bsz = x.shape[0]
        src_len = x_len + y_len
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),  ###xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
            value=True,
        )
        y_attn_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = (
            torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            .unsqueeze(0)
            .expand(bsz * self.num_head, -1, -1)
            .view(bsz, self.num_head, src_len, src_len)
            .to(device=x.device, dtype=torch.bool)
        )

        idx = 0
        top_k = int(top_k)

        xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)

        logits = self.ar_predict_layer(xy_dec[:, -1])
        logits = logits[:, :-1]
        samples = sample(logits, y, top_k=top_k, top_p=1, repetition_penalty=1.35, temperature=1.0)[0]
        y = torch.concat([y, samples], dim=1)
        y_emb = self.ar_audio_embedding(y[:, -1:])
        xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
            :, y_len + idx
        ].to(dtype=y_emb.dtype, device=y_emb.device)

        stop = False
        # for idx in range(1, 50):
        for idx in range(1, 1500):
            # [1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
            # y, k, v, y_emb, logits, samples = self.stage_decoder(y, k, v, y_emb, x_example)
            xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache)
            logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1]

            samples = sample(logits, y, top_k=top_k, top_p=1, repetition_penalty=1.35, temperature=1.0)[0]

            y = torch.concat([y, samples], dim=1)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                break

            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
                :, y_len + idx
            ].to(dtype=y_emb.dtype, device=y_emb.device)

        y[0, -1] = 0

        return y[:, -idx:].unsqueeze(0)


bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
cnhubert.cnhubert_base_path = cnhubert_base_path


@torch.jit.script
def build_phone_level_feature(res: Tensor, word2ph: IntTensor):
    phone_level_feature = []
    for i in range(word2ph.shape[0]):
        repeat_feature = res[i].repeat(word2ph[i].item(), 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # [sum(word2ph), 1024]
    return phone_level_feature


class MyBertModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(MyBertModel, self).__init__()
        self.bert = bert_model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, word2ph: IntTensor
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # res = torch.cat(outputs["hidden_states"][-3:-2], -1)[0][1:-1]
        res = torch.cat(outputs[1][-3:-2], -1)[0][1:-1]
        return build_phone_level_feature(res, word2ph)


class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl = cnhubert.get_model().model

    def forward(self, ref_audio_16k) -> torch.Tensor:
        ssl_content = self.ssl(ref_audio_16k)["last_hidden_state"].transpose(1, 2)
        return ssl_content


class ExportSSLModel(torch.nn.Module):
    def __init__(self, ssl: SSLModel):
        super().__init__()
        self.ssl = ssl

    def forward(self, ref_audio: torch.Tensor):
        return self.ssl(ref_audio)

    @torch.jit.export
    def resample(self, ref_audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
        audio = resamplex(ref_audio, src_sr, dst_sr).float()
        return audio


def export_bert(output_path):
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    text = "叹息声一声接着一声传出,木兰对着房门织布.听不见织布机织布的声音,只听见木兰在叹息.问木兰在想什么?问木兰在惦记什么?木兰答道,我也没有在想什么,也没有在惦记什么."
    ref_bert_inputs = tokenizer(text, return_tensors="pt")
    word2ph = []
    for c in text:
        if c in ["，", "。", "：", "？", ",", ".", "?"]:
            word2ph.append(1)
        else:
            word2ph.append(2)
    ref_bert_inputs["word2ph"] = torch.Tensor(word2ph).int()

    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path, output_hidden_states=True, torchscript=True)
    my_bert_model = MyBertModel(bert_model)

    ref_bert_inputs = {
        "input_ids": ref_bert_inputs["input_ids"],
        "attention_mask": ref_bert_inputs["attention_mask"],
        "token_type_ids": ref_bert_inputs["token_type_ids"],
        "word2ph": ref_bert_inputs["word2ph"],
    }

    torch._dynamo.mark_dynamic(ref_bert_inputs["input_ids"], 1)
    torch._dynamo.mark_dynamic(ref_bert_inputs["attention_mask"], 1)
    torch._dynamo.mark_dynamic(ref_bert_inputs["token_type_ids"], 1)
    torch._dynamo.mark_dynamic(ref_bert_inputs["word2ph"], 0)

    my_bert_model = torch.jit.trace(my_bert_model, example_kwarg_inputs=ref_bert_inputs)
    output_path = os.path.join(output_path, "bert_model.pt")
    my_bert_model.save(output_path)
    print("#### exported bert ####")


def export(gpt_path, vits_path, ref_audio_path, ref_text, output_path, export_bert_and_ssl=False, device="cpu"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"目录已创建: {output_path}")
    else:
        print(f"目录已存在: {output_path}")

    ref_audio = torch.tensor([load_audio(ref_audio_path, 16000)]).float()
    ssl = SSLModel()
    if export_bert_and_ssl:
        s = ExportSSLModel(torch.jit.trace(ssl, example_inputs=(ref_audio)))
        ssl_path = os.path.join(output_path, "ssl_model.pt")
        torch.jit.script(s).save(ssl_path)
        print("#### exported ssl ####")
        export_bert(output_path)
    else:
        s = ExportSSLModel(ssl)

    print(f"device: {device}")

    ref_seq_id, ref_bert_T, ref_norm_text = get_phones_and_bert(ref_text, "all_zh", "v2")
    ref_seq = torch.LongTensor([ref_seq_id]).to(device)
    ref_bert = ref_bert_T.T.to(ref_seq.device)
    text_seq_id, text_bert_T, norm_text = get_phones_and_bert(
        "这是一条测试语音，说什么无所谓，只是给它一个例子", "all_zh", "v2"
    )
    text_seq = torch.LongTensor([text_seq_id]).to(device)
    text_bert = text_bert_T.T.to(text_seq.device)

    ssl_content = ssl(ref_audio).to(device)

    # vits_path = "SoVITS_weights_v2/xw_e8_s216.pth"
    vits = VitsModel(vits_path).to(device)
    vits.eval()

    # gpt_path = "GPT_weights_v2/xw-e15.ckpt"
    # dict_s1 = torch.load(gpt_path, map_location=device)
    dict_s1 = torch.load(gpt_path)
    raw_t2s = get_raw_t2s_model(dict_s1).to(device)
    print("#### get_raw_t2s_model ####")
    print(raw_t2s.config)
    t2s_m = T2SModel(raw_t2s)
    t2s_m.eval()
    t2s = torch.jit.script(t2s_m).to(device)
    print("#### script t2s_m ####")

    print("vits.hps.data.sampling_rate:", vits.hps.data.sampling_rate)
    gpt_sovits = GPT_SoVITS(t2s, vits).to(device)
    gpt_sovits.eval()

    ref_audio_sr = s.resample(ref_audio, 16000, 32000).to(device)

    torch._dynamo.mark_dynamic(ssl_content, 2)
    torch._dynamo.mark_dynamic(ref_audio_sr, 1)
    torch._dynamo.mark_dynamic(ref_seq, 1)
    torch._dynamo.mark_dynamic(text_seq, 1)
    torch._dynamo.mark_dynamic(ref_bert, 0)
    torch._dynamo.mark_dynamic(text_bert, 0)

    top_k = torch.LongTensor([5]).to(device)

    with torch.no_grad():
        gpt_sovits_export = torch.jit.trace(
            gpt_sovits, example_inputs=(ssl_content, ref_audio_sr, ref_seq, text_seq, ref_bert, text_bert, top_k)
        )

        gpt_sovits_path = os.path.join(output_path, "gpt_sovits_model.pt")
        gpt_sovits_export.save(gpt_sovits_path)
        print("#### exported gpt_sovits ####")


@torch.jit.script
def parse_audio(ref_audio):
    ref_audio_16k = torchaudio.functional.resample(ref_audio, 48000, 16000).float()  # .to(ref_audio.device)
    ref_audio_sr = torchaudio.functional.resample(ref_audio, 48000, 32000).float()  # .to(ref_audio.device)
    return ref_audio_16k, ref_audio_sr


@torch.jit.script
def resamplex(ref_audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    return torchaudio.functional.resample(ref_audio, src_sr, dst_sr).float()


class GPT_SoVITS(nn.Module):
    def __init__(self, t2s: T2SModel, vits: VitsModel):
        super().__init__()
        self.t2s = t2s
        self.vits = vits

    def forward(
        self,
        ssl_content: torch.Tensor,
        ref_audio_sr: torch.Tensor,
        ref_seq: Tensor,
        text_seq: Tensor,
        ref_bert: Tensor,
        text_bert: Tensor,
        top_k: LongTensor,
        speed=1.0,
    ):
        codes = self.vits.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompts = prompt_semantic.unsqueeze(0)

        pred_semantic = self.t2s(prompts, ref_seq, text_seq, ref_bert, text_bert, top_k)
        audio = self.vits(text_seq, pred_semantic, ref_audio_sr, speed)
        return audio


def test():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument("--gpt_model", required=True, help="Path to the GPT model file")
    parser.add_argument("--sovits_model", required=True, help="Path to the SoVITS model file")
    parser.add_argument("--ref_audio", required=True, help="Path to the reference audio file")
    parser.add_argument("--ref_text", required=True, help="Path to the reference text file")
    parser.add_argument("--output_path", required=True, help="Path to the output directory")

    args = parser.parse_args()
    gpt_path = args.gpt_model
    vits_path = args.sovits_model
    ref_audio_path = args.ref_audio
    ref_text = args.ref_text

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    # bert_model = AutoModelForMaskedLM.from_pretrained(bert_path,output_hidden_states=True,torchscript=True)
    # bert = MyBertModel(bert_model)
    my_bert = torch.jit.load("onnx/bert_model.pt", map_location="cuda")

    # dict_s1 = torch.load(gpt_path, map_location="cuda")
    # raw_t2s = get_raw_t2s_model(dict_s1)
    # t2s = T2SModel(raw_t2s)
    # t2s.eval()
    # t2s = torch.jit.load("onnx/xw/t2s_model.pt",map_location='cuda')

    # vits_path = "SoVITS_weights_v2/xw_e8_s216.pth"
    # vits = VitsModel(vits_path)
    # vits.eval()

    # ssl = ExportSSLModel(SSLModel()).to('cuda')
    # ssl.eval()
    ssl = torch.jit.load("onnx/by/ssl_model.pt", map_location="cuda")

    # gpt_sovits = GPT_SoVITS(t2s,vits)
    gpt_sovits = torch.jit.load("onnx/by/gpt_sovits_model.pt", map_location="cuda")

    ref_seq_id, ref_bert_T, ref_norm_text = get_phones_and_bert(ref_text, "all_zh", "v2")
    ref_seq = torch.LongTensor([ref_seq_id])
    ref_bert = ref_bert_T.T.to(ref_seq.device)
    # text_seq_id,text_bert_T,norm_text = get_phones_and_bert("昨天晚上看见征兵文书,知道君主在大规模征兵,那么多卷征兵文册,每一卷上都有父亲的名字.","all_zh",'v2')
    text = "昨天晚上看见征兵文书,知道君主在大规模征兵,那么多卷征兵文册,每一卷上都有父亲的名字."

    text_seq_id, text_bert_T, norm_text = get_phones_and_bert(text, "all_zh", "v2")

    test_bert = tokenizer(text, return_tensors="pt")
    word2ph = []
    for c in text:
        if c in ["，", "。", "：", "？", "?", ",", "."]:
            word2ph.append(1)
        else:
            word2ph.append(2)
    test_bert["word2ph"] = torch.Tensor(word2ph).int()

    test_bert = my_bert(
        test_bert["input_ids"].to("cuda"),
        test_bert["attention_mask"].to("cuda"),
        test_bert["token_type_ids"].to("cuda"),
        test_bert["word2ph"].to("cuda"),
    )

    text_seq = torch.LongTensor([text_seq_id])
    text_bert = text_bert_T.T.to(text_seq.device)

    print("text_bert:", text_bert.shape, text_bert)
    print("test_bert:", test_bert.shape, test_bert)
    print(torch.allclose(text_bert.to("cuda"), test_bert))

    print("text_seq:", text_seq.shape)
    print("text_bert:", text_bert.shape, text_bert.type())

    # [1,N]
    ref_audio = torch.tensor([load_audio(ref_audio_path, 16000)]).float().to("cuda")
    print("ref_audio:", ref_audio.shape)

    ref_audio_sr = ssl.resample(ref_audio, 16000, 32000)
    print("start ssl")
    ssl_content = ssl(ref_audio)

    print("start gpt_sovits:")
    print("ssl_content:", ssl_content.shape)
    print("ref_audio_sr:", ref_audio_sr.shape)
    print("ref_seq:", ref_seq.shape)
    ref_seq = ref_seq.to("cuda")
    print("text_seq:", text_seq.shape)
    text_seq = text_seq.to("cuda")
    print("ref_bert:", ref_bert.shape)
    ref_bert = ref_bert.to("cuda")
    print("text_bert:", text_bert.shape)
    text_bert = text_bert.to("cuda")

    top_k = torch.LongTensor([5]).to("cuda")

    with torch.no_grad():
        audio = gpt_sovits(ssl_content, ref_audio_sr, ref_seq, text_seq, ref_bert, test_bert, top_k)
    print("start write wav")
    soundfile.write("out.wav", audio.detach().cpu().numpy(), 32000)


import text
import json


def export_symbel(version="v2"):
    if version == "v1":
        symbols = text._symbol_to_id_v1
        with open("onnx/symbols_v1.json", "w") as file:
            json.dump(symbols, file, indent=4)
    else:
        symbols = text._symbol_to_id_v2
        with open("onnx/symbols_v2.json", "w") as file:
            json.dump(symbols, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument("--gpt_model", required=True, help="Path to the GPT model file")
    parser.add_argument("--sovits_model", required=True, help="Path to the SoVITS model file")
    parser.add_argument("--ref_audio", required=True, help="Path to the reference audio file")
    parser.add_argument("--ref_text", required=True, help="Path to the reference text file")
    parser.add_argument("--output_path", required=True, help="Path to the output directory")
    parser.add_argument("--export_common_model", action="store_true", help="Export Bert and SSL model")
    parser.add_argument("--device", help="Device to use")

    args = parser.parse_args()
    export(
        gpt_path=args.gpt_model,
        vits_path=args.sovits_model,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output_path,
        device=args.device,
        export_bert_and_ssl=args.export_common_model,
    )


import inference_webui

if __name__ == "__main__":
    inference_webui.is_half = False
    inference_webui.dtype = torch.float32
    main()
    # test()
