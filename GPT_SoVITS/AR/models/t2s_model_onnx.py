# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import math
from typing import List, Optional
import torch
from tqdm import tqdm

from AR.models.utils import (
    sample,
)
from AR.modules.embedding import TokenEmbedding
from AR.modules.transformer import LayerNorm
from AR.modules.transformer import TransformerEncoder
from AR.modules.transformer import TransformerEncoderLayer
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

from torch.distributions import Exponential

ISONNXEXPORT = False

default_config = {
    "model": {
        "vocab_size": 1025,
        "phoneme_vocab_size": 512,
        "embedding_dim": 1024,
        "hidden_dim": 1024,
        "head": 16,
        "linear_units": 2048,
        "n_layer": 16,
        "dropout": 0,
        "EOS": 1024,
    }
}

def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = -torch.log(torch.rand_like(probs_sort)) #https://github.com/RVC-Boss/GPT-SoVITS/pull/835
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.long)

def logits_to_probs(
    logits,
    previous_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor
):
    # if previous_tokens is not None:
    #     previous_tokens = previous_tokens.squeeze()
    # print(logits.shape,previous_tokens.shape)
    # pdb.set_trace()
    previous_tokens = previous_tokens.long()
    score = torch.gather(logits, dim=1, index=previous_tokens)
    score = torch.where(
        score < 0, score * repetition_penalty, score / repetition_penalty
    )
    logits.scatter_(dim=1, index=previous_tokens, src=score)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
    )
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[:, 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / torch.clamp_min(temperature, 1e-5)

    v, _ = torch.topk(logits, top_k)
    pivot = v[: , -1].unsqueeze(-1)
    logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
):
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

# @torch.jit.script ## 使用的话首次推理会非常慢，而且推理速度不稳定
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, attn_mask:Optional[torch.Tensor]=None, scale:Optional[torch.Tensor]=None) -> torch.Tensor:
    B, H, L, S =query.size(0), query.size(1), query.size(-2), key.size(-2)
    if scale is None:
        scale_factor = torch.tensor(1 / math.sqrt(query.size(-1)))
    else:
        scale_factor = scale
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(attn_mask, float("-inf"))
        else:
            attn_bias = attn_bias + attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(attn_mask, 0)
        else:
            attn_mask = attn_mask.clone()
            attn_mask[attn_mask!=float("-inf")] =0
            attn_mask[attn_mask==float("-inf")] =1
            attn_weight = attn_weight.masked_fill(attn_mask, 0)

    return attn_weight @ value

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
            num_heads,
            hidden_dim: int,
            mlp: T2SMLP,
            qkv_w,
            qkv_b,
            out_w,
            out_b,
            norm_w1,
            norm_b1,
            norm_eps1,
            norm_w2,
            norm_b2,
            norm_eps2,
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
    def to_mask(self, x:torch.Tensor, padding_mask:Optional[torch.Tensor]):
        if padding_mask is None:
            return x
        
        if padding_mask.dtype == torch.bool:
            return x.masked_fill(padding_mask, 0)
        else:
            return x * padding_mask
        
    def process_prompt(self, x:torch.Tensor, attn_mask : torch.Tensor, padding_mask:Optional[torch.Tensor]=None, torch_sdpa:bool=True):
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

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(self.to_mask(attn, padding_mask), self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(
            x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache
    
    def decode_next_token(self, x, k_cache, v_cache):
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

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(
            x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
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
    def __init__(self, num_blocks : int, blocks: List[T2SBlock]):
        self.num_blocks : int = num_blocks
        self.blocks = blocks

    def process_prompt(
        self, x:torch.Tensor, attn_mask : torch.Tensor,
        padding_mask : Optional[torch.Tensor]=None, 
        torch_sdpa:bool=True
        ):
        k_cache : List[torch.Tensor] = []
        v_cache : List[torch.Tensor] = []
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(x, attn_mask, padding_mask, torch_sdpa)
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache

    def decode_next_token(
        self, x:torch.Tensor, 
        k_cache, 
        v_cache, 
    ):
        K_Cache = []
        V_Cache = []
        for i in range(self.num_blocks):
            x, k, v = self.blocks[i].decode_next_token(x, k_cache[i], v_cache[i])
            K_Cache.append(k)
            V_Cache.append(v)
        K_Cache = torch.stack(K_Cache, dim=0)
        V_Cache = torch.stack(V_Cache, dim=0)
        return x, K_Cache, V_Cache
    

class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 114514))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.embedding_dim)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor, x_size) -> torch.Tensor:
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output[:,:x_size,:] = output[:,:x_size,:] * self.x_scale + self.alpha * self.pe[:, : x_size]
        return self.dropout(output)


class PromptProcessor(nn.Module):
    def __init__(self, cache_len, model, top_k):
        super(PromptProcessor, self).__init__()
        self.top_k = top_k
        self.model = model
        self.ar_text_embedding = model.ar_text_embedding
        self.ar_text_position = model.ar_text_position
        self.ar_audio_embedding = model.ar_audio_embedding
        self.ar_audio_position = model.ar_audio_position
        self.bert_proj = model.bert_proj
        cache_len = torch.tensor(cache_len)
        self.register_buffer("cache_len", cache_len, persistent=False)

    def forward(self, x, x_len, y, y_len, bert_feature, top_p, repetition_penalty, temperature):
        bsz = x.size(0)
        src_len = x_len + y_len

        x_emb = self.ar_text_embedding(x)
        x_emb = x_emb + self.bert_proj(bert_feature)
        x_pos = self.ar_text_position(x_emb, x_len)
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb, y_len)
        y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),(x_len, 0),value=False)

        xy_pos = torch.concat([x_pos, y_pos], dim=1)

        x_attn_mask_pad = F.pad(x_attn_mask,(0, y_len),value=True)
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).unsqueeze(0)\
                                                .expand(bsz * self.model.num_head, -1, -1)\
                                                .view(bsz, self.model.num_head, src_len, src_len)\
                                                .to(device=x.device, dtype=torch.bool)

        xy_dec, k_cache, v_cache = self.model.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)
        
        logits = self.model.ar_predict_layer(
                xy_dec[:, -1]
            )

        samples = sample(
            logits, y, top_k=self.top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
        )[0]
        y = torch.concat([y, samples], dim=1)

        y_emb = self.ar_audio_embedding(samples)
        xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len].to(dtype=y_emb.dtype,device=y_emb.device)
        k_cache = torch.stack(k_cache, dim=0)
        v_cache = torch.stack(v_cache, dim=0)
        return y, k_cache, v_cache, xy_pos, y_len + 1, samples
    

class DecodeNextToken(nn.Module):
    def __init__(self, cache_len, model, top_k):
        super(DecodeNextToken, self).__init__()
        self.top_k = top_k
        self.model = model
        self.ar_text_embedding = model.ar_text_embedding
        self.ar_text_position = model.ar_text_position
        self.ar_audio_embedding = model.ar_audio_embedding
        self.ar_audio_position = model.ar_audio_position
        cache_len = torch.tensor(cache_len)
        self.register_buffer("cache_len", cache_len, persistent=False)

    def forward(self, y, k_cache, v_cache, xy_pos, y_idx, top_p, repetition_penalty, temperature):
        xy_dec, k_cache, v_cache = self.model.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache)
        logits = self.model.ar_predict_layer(
                xy_dec[:, -1]
            )

        samples = sample(
            logits, y, top_k=self.top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
        )[0]
        y = torch.concat([y, samples], dim=1)

        y_emb = self.ar_audio_embedding(samples)
        xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_idx].to(dtype=y_emb.dtype,device=y_emb.device)

        return y, k_cache, v_cache, xy_pos, y_idx + 1, samples
    


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim, self.phoneme_vocab_size, self.p_dropout
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim, self.vocab_size, self.p_dropout
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True
        )

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )

        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )

        blocks = []

        for i in range(self.num_layers):
            layer = self.h.layers[i]
            t2smlp = T2SMLP(
                layer.linear1.weight,
                layer.linear1.bias,
                layer.linear2.weight,
                layer.linear2.bias
            )

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
                layer.norm2.eps
            )

            blocks.append(block)
        
        self.t2s_transformer = T2STransformer(self.num_layers, blocks)

    def infer_panel_naive(
        self,
        x:torch.LongTensor,  #####全部文本token
        x_lens:torch.LongTensor,
        prompts:torch.LongTensor,  ####参考音频token
        bert_feature:torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        y = prompts

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        k_cache = None
        v_cache = None

        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        bsz = x.shape[0]
        src_len = x_len + y_len
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)\
                                                .unsqueeze(0)\
                                                .expand(bsz*self.num_head, -1, -1)\
                                                .view(bsz, self.num_head, src_len, src_len)\
                                                .to(device=x.device, dtype=torch.bool)

        for idx in tqdm(range(1500)):
            if xy_attn_mask is not None:
                xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)
            else:
                xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache)

            logits = self.ar_predict_layer(
                xy_dec[:, -1]
            )

            if idx == 0:
                xy_attn_mask = None
            if(idx<11):###至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1]

            samples = sample(
                logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
            )[0]

            y = torch.concat([y, samples], dim=1)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            ####################### update next step ###################################
            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to(dtype=y_emb.dtype,device=y_emb.device)

        return y[:, :-1], idx
    
    def infer(self, x, prompts, bert_feature):
        top_k = self.top_k
        early_stop_num = self.early_stop_num

        x = self.onnx_encoder(x, bert_feature)

        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_example = x[:,:,0] * 0.0
        x_attn_mask = torch.matmul(x_example.transpose(0, 1), x_example)
        x_attn_mask = torch.zeros_like(x_attn_mask, dtype=torch.bool)

        stop = False
        cache = {
            "all_stage": self.num_layers,
            "k": [None] * self.num_layers,
            "v": [None] * self.num_layers,
            "y_emb": None,
            "first_infer": 1,
            "stage": 0,
        }
        for idx in range(1500):
            if cache["first_infer"] == 1:
                y_emb = self.ar_audio_embedding(y)
            else:
                y_emb = torch.cat(
                    [cache["y_emb"], self.ar_audio_embedding(y[:, -1:])], 1
                )
            cache["y_emb"] = y_emb
            y_pos = self.ar_audio_position(y_emb)
            if cache["first_infer"] == 1:
                xy_pos = torch.concat([x, y_pos], dim=1)
            else:
                xy_pos = y_pos[:, -1:]
            y_len = y_pos.shape[1]
            if cache["first_infer"] == 1:
                x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
                y_attn_mask = F.pad(
                    torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                    (x_len, 0), value=False
                )
                xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            else:
                xy_attn_mask = torch.zeros((1, x_len + y_len), dtype=torch.bool)
            xy_dec = self.h(xy_pos, mask=xy_attn_mask, cache=cache)
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = sample(logits[0], y, top_k=top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                break
            y = torch.concat([y, samples], dim=1)
            cache["first_infer"] = 0
        return y, idx