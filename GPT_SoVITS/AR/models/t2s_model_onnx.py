# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/model/t2s_model.py
import torch
from tqdm import tqdm

from AR.modules.embedding_onnx import SinePositionalEmbedding
from AR.modules.embedding_onnx import TokenEmbedding
from AR.modules.transformer_onnx import LayerNorm
from AR.modules.transformer_onnx import TransformerEncoder
from AR.modules.transformer_onnx import TransformerEncoderLayer
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

inf_tensor_value = torch.FloatTensor([-float("Inf")]).float()

def logits_to_probs(
    logits,
    previous_tokens = None,
    temperature: float = 1.0,
    top_k = None,
    top_p = None,
    repetition_penalty: float = 1.0,
):
    previous_tokens = previous_tokens.squeeze()
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, inf_tensor_value, logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync(
    probs_sort
):  # Does multinomial sampling without a cuda synchronization
    q = torch.randn_like(probs_sort)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits,
    previous_tokens,
    **sampling_kwargs,
):
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


class OnnxEncoder(nn.Module):
    def __init__(self, ar_text_embedding, bert_proj, ar_text_position):
        super().__init__()
        self.ar_text_embedding = ar_text_embedding
        self.bert_proj = bert_proj
        self.ar_text_position = ar_text_position
    
    def forward(self, x, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        return self.ar_text_position(x)


class T2SFirstStageDecoder(nn.Module):
    def __init__(self, ar_audio_embedding, ar_audio_position, h, ar_predict_layer, loss_fct, ar_accuracy_metric,
    top_k, early_stop_num, num_layers):
        super().__init__()
        self.ar_audio_embedding = ar_audio_embedding
        self.ar_audio_position = ar_audio_position
        self.h = h
        self.ar_predict_layer = ar_predict_layer
        self.loss_fct = loss_fct
        self.ar_accuracy_metric = ar_accuracy_metric
        self.top_k = top_k
        self.early_stop_num = early_stop_num
        self.num_layers = num_layers
    
    def forward(self, x, prompt):
        y = prompt
        x_example = x[:,:,0] * 0.0
        #N, 1, 512
        cache = {
            "all_stage": self.num_layers,
            "k": None,
            "v": None,
            "y_emb": None,
            "first_infer": 1,
            "stage": 0,
        }

        y_emb = self.ar_audio_embedding(y)

        cache["y_emb"] = y_emb
        y_pos = self.ar_audio_position(y_emb)

        xy_pos = torch.concat([x, y_pos], dim=1)

        y_example = y_pos[:,:,0] * 0.0
        x_attn_mask = torch.matmul(x_example.transpose(0, 1) , x_example).bool()
        y_attn_mask = torch.ones_like(torch.matmul(y_example.transpose(0, 1), y_example), dtype=torch.int64)
        y_attn_mask = torch.cumsum(y_attn_mask, dim=1) - torch.cumsum(
            torch.ones_like(y_example.transpose(0, 1), dtype=torch.int64), dim=0
        )
        y_attn_mask = y_attn_mask > 0

        x_y_pad = torch.matmul(x_example.transpose(0, 1), y_example).bool()
        y_x_pad = torch.matmul(y_example.transpose(0, 1), x_example).bool()
        x_attn_mask_pad = torch.cat([x_attn_mask, torch.ones_like(x_y_pad)], dim=1)
        y_attn_mask = torch.cat([y_x_pad, y_attn_mask], dim=1)
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
        cache["k"] = torch.matmul(x_attn_mask_pad[0].float().unsqueeze(-1), torch.zeros((1, 512)))\
        .unsqueeze(1).repeat(self.num_layers, 1, 1, 1)
        cache["v"] = torch.matmul(x_attn_mask_pad[0].float().unsqueeze(-1), torch.zeros((1, 512)))\
        .unsqueeze(1).repeat(self.num_layers, 1, 1, 1)

        xy_dec = self.h(xy_pos, mask=xy_attn_mask, cache=cache)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits[0], y, top_k=self.top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)

        y = torch.concat([y, samples], dim=1)

        return y, cache["k"], cache["v"], cache["y_emb"], x_example


class T2SStageDecoder(nn.Module):
    def __init__(self, ar_audio_embedding, ar_audio_position, h, ar_predict_layer, loss_fct, ar_accuracy_metric,
    top_k, early_stop_num, num_layers):
        super().__init__()
        self.ar_audio_embedding = ar_audio_embedding
        self.ar_audio_position = ar_audio_position
        self.h = h
        self.ar_predict_layer = ar_predict_layer
        self.loss_fct = loss_fct
        self.ar_accuracy_metric = ar_accuracy_metric
        self.top_k = top_k
        self.early_stop_num = early_stop_num
        self.num_layers = num_layers

    def forward(self, y, k, v, y_emb, x_example):
        cache = {
            "all_stage": self.num_layers,
            "k": torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 1)),
            "v": torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 1)),
            "y_emb": y_emb,
            "first_infer": 0,
            "stage": 0,
        }

        y_emb = torch.cat(
            [cache["y_emb"], self.ar_audio_embedding(y[:, -1:])], 1
        )
        cache["y_emb"] = y_emb
        y_pos = self.ar_audio_position(y_emb)

        xy_pos = y_pos[:, -1:]
        
        y_example = y_pos[:,:,0] * 0.0

        xy_attn_mask = torch.cat([x_example, y_example], dim=1)
        xy_attn_mask = torch.zeros_like(xy_attn_mask, dtype=torch.bool)

        xy_dec = self.h(xy_pos, mask=xy_attn_mask, cache=cache)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits[0], y, top_k=self.top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)

        y = torch.concat([y, samples], dim=1)

        return y, cache["k"], cache["v"], cache["y_emb"], logits, samples


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
        self.p_dropout = float(config["model"]["dropout"])
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
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
        self.top_k = torch.LongTensor([1])
        self.early_stop_num = torch.LongTensor([-1])

    def init_onnx(self):
        self.onnx_encoder = OnnxEncoder(self.ar_text_embedding, self.bert_proj, self.ar_text_position)
        self.first_stage_decoder = T2SFirstStageDecoder(self.ar_audio_embedding, self.ar_audio_position, self.h, 
            self.ar_predict_layer, self.loss_fct, self.ar_accuracy_metric, self.top_k, self.early_stop_num,
            self.num_layers)
        self.stage_decoder = T2SStageDecoder(self.ar_audio_embedding, self.ar_audio_position, self.h, 
            self.ar_predict_layer, self.loss_fct, self.ar_accuracy_metric, self.top_k, self.early_stop_num,
            self.num_layers)

    def forward(self, x, prompts, bert_feature):
        early_stop_num = self.early_stop_num
        prefix_len = prompts.shape[1]

        x = self.onnx_encoder(x, bert_feature)
        y, k, v, y_emb, stage, x_example = self.first_stage_decoder(x, prompts)

        stop = False
        for idx in range(1, 1500):
            enco = self.stage_decoder(y, k, v, y_emb, stage, x_example)
            y, k, v, y_emb, stage, logits, samples = enco
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0
        return y, idx

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