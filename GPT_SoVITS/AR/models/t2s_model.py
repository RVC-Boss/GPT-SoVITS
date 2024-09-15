# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import math
from typing import List, Optional
import torch
from tqdm import tqdm

from AR.models.utils import make_pad_mask
from AR.models.utils import (
    topk_sampling,
    sample,
    logits_to_probs,
    multinomial_sample_one_no_sync,
    dpo_loss,
    make_reject_y,
    get_batch_logps
)
from AR.modules.embedding import SinePositionalEmbedding
from AR.modules.embedding import TokenEmbedding
from AR.modules.transformer import LayerNorm
from AR.modules.transformer import TransformerEncoder
from AR.modules.transformer import TransformerEncoderLayer
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

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
            attn_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(attn_mask, 0)
        else:
            attn_mask[attn_mask!=float("-inf")] =0
            attn_mask[attn_mask==float("-inf")] =1
            attn_weight.masked_fill_(attn_mask, 0)

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

        attn = attn.permute(2, 0, 1, 3).reshape(batch_size*q_len, self.hidden_dim)
        attn = attn.view(q_len, batch_size, self.hidden_dim).transpose(1, 0)
        attn = F.linear(self.to_mask(attn, padding_mask), self.out_w, self.out_b)

        if padding_mask is not None:
            for i in range(batch_size):
                # mask = padding_mask[i,:,0]
                if self.false.device!= padding_mask.device:
                    self.false = self.false.to(padding_mask.device)
                idx = torch.where(padding_mask[i,:,0]==self.false)[0]
                x_item = x[i,idx,:].unsqueeze(0)
                attn_item = attn[i,idx,:].unsqueeze(0)
                x_item = x_item + attn_item
                x_item = F.layer_norm(
                    x_item, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
                )
                x_item = x_item + self.mlp.forward(x_item)
                x_item = F.layer_norm(
                    x_item,
                    [self.hidden_dim],
                    self.norm_w2,
                    self.norm_b2,
                    self.norm_eps2,
                )
                x[i,idx,:] = x_item.squeeze(0)
            x = self.to_mask(x, padding_mask)
        else:
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
    
    def decode_next_token(self, x:torch.Tensor, k_cache:torch.Tensor, v_cache:torch.Tensor, attn_mask:Optional[torch.Tensor]=None, torch_sdpa:bool=True):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        k_cache = torch.cat([k_cache, k], dim=1)
        v_cache = torch.cat([v_cache, v], dim=1)
        
        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k_cache.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)


        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.permute(2, 0, 1, 3).reshape(batch_size*q_len, self.hidden_dim)
        attn = attn.view(q_len, batch_size, self.hidden_dim).transpose(1, 0)
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
        k_cache: List[torch.Tensor], 
        v_cache: List[torch.Tensor], 
        attn_mask : Optional[torch.Tensor]=None,
        torch_sdpa:bool=True
    ):
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token(x, k_cache[i], v_cache[i], attn_mask, torch_sdpa)
        return x, k_cache, v_cache


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
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
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

    def make_input_data(self, x, x_lens, y, y_lens, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)

        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        # x_attn_mask[:, x_len]=False
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )

        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)

        return xy_pos, xy_attn_mask, targets

    def forward(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """

        reject_y, reject_y_lens = make_reject_y(y, y_lens)

        xy_pos, xy_attn_mask, targets = self.make_input_data(x, x_lens, y, y_lens, bert_feature)

        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        x_len = x_lens.max()
        logits = self.ar_predict_layer(xy_dec[:, x_len:])

        ###### DPO #############
        reject_xy_pos, reject_xy_attn_mask, reject_targets = self.make_input_data(x, x_lens, reject_y, reject_y_lens, bert_feature)

        reject_xy_dec, _ = self.h(
            (reject_xy_pos, None),
            mask=reject_xy_attn_mask,
        )
        x_len = x_lens.max()
        reject_logits = self.ar_predict_layer(reject_xy_dec[:, x_len:])

        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum

        loss_1 = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.permute(0, 2, 1).detach(), targets).item()

        A_logits, R_logits = get_batch_logps(logits, reject_logits, targets, reject_targets)
        loss_2, _, _ = dpo_loss(A_logits, R_logits, 0, 0, 0.2, reference_free=True)
        
        loss = loss_1 + loss_2

        return loss, acc

    def forward_old(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
        loss = F.cross_entropy(logits, targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        return loss, acc

    # 需要看下这个函数和 forward 的区别以及没有 semantic 的时候 prompts 输入什么
    def infer(
            self,
            x,
            x_lens,
            prompts,
            bert_feature,
            top_k: int = -100,
            early_stop_num: int = -1,
            temperature: float = 1.0,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        for _ in tqdm(range(1500)):
            y_emb = self.ar_audio_embedding(y)
            y_pos = self.ar_audio_position(y_emb)
            # x 和逐渐增长的 y 一起输入给模型
            xy_pos = torch.concat([x, y_pos], dim=1)
            y_len = y.shape[1]
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
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(
                y.device
            )

            xy_dec, _ = self.h(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            # print(samples.shape)#[1,1]#第一个1是bs
            # import os
            # os._exit(2333)
            y = torch.concat([y, samples], dim=1)
        return y

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
        )
        # 错位
        return targets[:, :-1], targets[:, 1:]

    def infer_panel_batch_infer(
        self,
        x:List[torch.LongTensor],  #####全部文本token
        x_lens:torch.LongTensor,
        prompts:torch.LongTensor,  ####参考音频token
        bert_feature:List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        if prompts is None:
            print("Warning: Prompt free is not supported batch_infer! switch to naive_infer")
            return self.infer_panel_naive_batched(x, x_lens, prompts, bert_feature, top_k=top_k, top_p=top_p, early_stop_num=early_stop_num, temperature=temperature, **kwargs)


        max_len = kwargs.get("max_len",x_lens.max())
        x_list = []
        for x_item, bert_item in zip(x, bert_feature):
            # max_len = max(max_len, x_item.shape[0], bert_item.shape[1])
            x_item = self.ar_text_embedding(x_item.unsqueeze(0))
            x_item = x_item + self.bert_proj(bert_item.transpose(0, 1).unsqueeze(0))
            x_item = self.ar_text_position(x_item).squeeze(0)
            x_item = F.pad(x_item,(0,0,0,max_len-x_item.shape[0]),value=0) if x_item.shape[0]<max_len else x_item
            x_list.append(x_item)
        x = torch.stack(x_list, dim=0)


        # AR Decoder
        y = prompts
        
        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False

        k_cache = None
        v_cache = None
        ###################  first step ##########################
        if y is not None:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_lens = torch.LongTensor([y_emb.shape[1]]*y_emb.shape[0]).to(x.device)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_lens = torch.LongTensor([y_len]*x.shape[0]).to(x.device)
            y_pos = None
            xy_pos = x
            y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
            ref_free = True


        ##### create mask #####
        bsz = x.shape[0]
        src_len = x_len + y_len
        y_paddind_mask = make_pad_mask(y_lens, y_len)
        x_paddind_mask = make_pad_mask(x_lens, max_len)
        
        # (bsz, x_len + y_len)
        xy_padding_mask = torch.concat([x_paddind_mask, y_paddind_mask], dim=1)

        x_mask = F.pad(
            x_attn_mask,
            (0, y_len),  ###xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
            value=True,
        )
        y_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), 
            (x_len, 0),
            value=False,
        )
        
        xy_mask = torch.concat([x_mask, y_mask], dim=0).view(1 , src_len, src_len).repeat(bsz, 1, 1).to(x.device)
        _xy_padding_mask = xy_padding_mask.view(bsz, 1, src_len).repeat(1, src_len, 1)
        
        for i in range(bsz):
            l = x_lens[i]
            _xy_padding_mask[i,l:max_len,:]=True
            
        xy_attn_mask = xy_mask.logical_or(_xy_padding_mask)
        xy_attn_mask = xy_attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1)
        xy_attn_mask = xy_attn_mask.bool()
        xy_padding_mask = xy_padding_mask.view(bsz, src_len, 1).expand(-1, -1, self.model_dim)

        ###### decode #####
        y_list = [None]*y.shape[0]
        batch_idx_map = list(range(y.shape[0]))
        idx_list = [None]*y.shape[0]
        for idx in tqdm(range(1500)):
            if idx == 0:
                xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, xy_padding_mask, False)
            else:
                xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache, xy_attn_mask, False)
            logits = self.ar_predict_layer(
                xy_dec[:, -1]
            )

            if idx == 0:
                xy_attn_mask = F.pad(xy_attn_mask[:,:,-1].unsqueeze(-2),(0,1),value=False)
                logits = logits[:, :-1]
            else:
                xy_attn_mask = F.pad(xy_attn_mask,(0,1),value=False)

            samples = sample(
                    logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
                )[0]

            y = torch.concat([y, samples], dim=1)
            
            ####### 移除batch中已经生成完毕的序列,进一步优化计算量
            tokens = torch.argmax(logits, dim=-1)
            reserved_idx_of_batch_for_y = None
            if (self.EOS in samples[:, 0]) or \
                (self.EOS in tokens):  ###如果生成到EOS，则停止
                    l1 = samples[:, 0]==self.EOS
                    l2 = tokens==self.EOS
                    l = l1.logical_or(l2)
                    removed_idx_of_batch_for_y = torch.where(l==True)[0].tolist()
                    reserved_idx_of_batch_for_y = torch.where(l==False)[0]
                    # batch_indexs = torch.tensor(batch_idx_map, device=y.device)[removed_idx_of_batch_for_y]
                    for i in removed_idx_of_batch_for_y:
                        batch_index = batch_idx_map[i]
                        idx_list[batch_index] = idx - 1
                        y_list[batch_index] = y[i, :-1]
                
                    batch_idx_map = [batch_idx_map[i] for i in reserved_idx_of_batch_for_y.tolist()]
                
            # 只保留batch中未生成完毕的序列 
            if reserved_idx_of_batch_for_y is not None:
                # index = torch.LongTensor(batch_idx_map).to(y.device)
                y = torch.index_select(y, dim=0, index=reserved_idx_of_batch_for_y)
                xy_attn_mask = torch.index_select(xy_attn_mask, dim=0, index=reserved_idx_of_batch_for_y)
                if k_cache is not None :
                    for i in range(len(k_cache)):
                        k_cache[i] = torch.index_select(k_cache[i], dim=0, index=reserved_idx_of_batch_for_y)
                        v_cache[i] = torch.index_select(v_cache[i], dim=0, index=reserved_idx_of_batch_for_y)
                
                
            if (early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num) or idx==1499:
                print("use early stop num:", early_stop_num)
                stop = True
                for i, batch_index in enumerate(batch_idx_map):
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]
                
            if not (None in idx_list):
                stop = True
                
            if stop:
                if y.shape[1]==0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            ####################### update next step ###################################
            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to( dtype= y_emb.dtype,device=y_emb.device)            

        if (None in idx_list):
            for i in range(x.shape[0]):
                if idx_list[i] is None:
                    idx_list[i] = 1500-1  ###如果没有生成到EOS，就用最大长度代替
                    
        if ref_free:
            return y_list, [0]*x.shape[0]
        # print(idx_list)
        return y_list, idx_list
    
    def infer_panel_naive_batched(self,
        x:List[torch.LongTensor],  #####全部文本token
        x_lens:torch.LongTensor,
        prompts:torch.LongTensor,  ####参考音频token
        bert_feature:List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs
        ):
        y_list = []
        idx_list = []
        for i in range(len(x)):
            y, idx = self.infer_panel_naive(x[i].unsqueeze(0), 
                                                  x_lens[i], 
                                                  prompts[i].unsqueeze(0) if prompts is not None else None, 
                                                  bert_feature[i].unsqueeze(0), 
                                                  top_k, 
                                                  top_p, 
                                                  early_stop_num, 
                                                  temperature,
                                                  repetition_penalty,
                                                  **kwargs)
            y_list.append(y[0])
            idx_list.append(idx)
        
        return y_list, idx_list
    
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

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        # print(1111111,self.num_layers)

        k_cache = None
        v_cache = None
        ###################  first step ##########################
        if y is not None:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
            ref_free = True

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

        if ref_free:
            return y[:, :-1], 0
        return y[:, :-1], idx - 1
    
    
    def infer_panel(
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
        return self.infer_panel_naive(x, x_lens, prompts, bert_feature, top_k, top_p, early_stop_num, temperature, repetition_penalty, **kwargs)
