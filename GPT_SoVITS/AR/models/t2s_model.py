# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import torch
import random
import numpy as np

from tqdm import tqdm
from typing import List
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

    def process_prompt(self, x, attn_mask: torch.Tensor):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]

        k_cache = k
        v_cache = v

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)

        attn = attn.permute(2, 0, 1, 3).reshape(batch_size, -1, self.hidden_dim)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = F.layer_norm(
            x + attn, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = F.layer_norm(
            x + self.mlp.forward(x),
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
        kv_len = k_cache.shape[1]

        batch_size = q.shape[0]
        q_len = q.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.permute(2, 0, 1, 3).reshape(batch_size, -1, self.hidden_dim)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = F.layer_norm(
            x + attn, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = F.layer_norm(
            x + self.mlp.forward(x),
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache


@torch.jit.script
class T2STransformer:
    def __init__(self, num_blocks: int, blocks: List[T2SBlock]):
        self.num_blocks: int = num_blocks
        self.blocks = blocks

    def process_prompt(
            self, x, attn_mask: torch.Tensor):
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(x, attn_mask)
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache

    def decode_next_token(
            self, x, k_cache: List[torch.Tensor], v_cache: List[torch.Tensor]
    ):
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token(x, k_cache[i], v_cache[i])
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
            # (layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
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

    def infer_panel(
            self,
            x,  #####全部文本token
            x_lens,
            prompts,  ####参考音频token
            bert_feature,
            top_k: int = -100,
            top_p: int = 100,
            early_stop_num: int = -1,
            temperature: float = 1.0,
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
            prompts = y
            ref_free = True

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
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(
            x.device
        )

        for idx in tqdm(range(1500)):
            if xy_attn_mask is not None:
                xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, xy_attn_mask)
            else:
                xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache)

            logits = self.ar_predict_layer(
                xy_dec[:, -1]
            )

            if idx == 0:
                xy_attn_mask = None
                logits = logits[:, :-1]
            samples = sample(
                logits[0], y, top_k=top_k, top_p=top_p, repetition_penalty=1.35, temperature=temperature
            )[0].unsqueeze(0)

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