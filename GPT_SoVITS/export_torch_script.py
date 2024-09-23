# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
from typing import Optional
from my_utils import load_audio
from onnx_export import VitsModel
from text import cleaned_text_to_sequence
import torch
import torchaudio

from torch import IntTensor, LongTensor, nn
from torch.nn import functional as F

from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert

from AR.models.t2s_lightning_module import Text2SemanticLightningModule


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
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=1, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
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

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v[: , -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def get_raw_t2s_model(dict_s1) -> Text2SemanticLightningModule: 
    config = dict_s1["config"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_model = t2s_model.eval()
    return t2s_model

def multinomial_sample_one_no_sync(
    probs_sort
):  # Does multinomial sampling without a cuda synchronization
    q = torch.randn_like(probs_sort)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits,
    previous_tokens,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


class T2SModel(nn.Module):
    def __init__(self, config,raw_t2s:Text2SemanticLightningModule, norm_first=False, top_k=3):
        super(T2SModel, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = float(config["model"]["dropout"])
        self.EOS:int = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        self.hz = 50
        self.config = config

        # self.bert_proj = nn.Linear(1024, self.embedding_dim)
        # self.ar_text_embedding = TokenEmbedding(
        #     self.embedding_dim, self.phoneme_vocab_size, self.p_dropout
        # )
        # self.ar_text_position = SinePositionalEmbedding(
        #     self.embedding_dim, dropout=0.1, scale=False, alpha=True
        # )
        # self.ar_audio_embedding = TokenEmbedding(
        #     self.embedding_dim, self.vocab_size, self.p_dropout
        # )
        # self.ar_audio_position = SinePositionalEmbedding(
        #     self.embedding_dim, dropout=0.1, scale=False, alpha=True
        # )

        self.bert_proj = raw_t2s.model.bert_proj
        self.ar_text_embedding = raw_t2s.model.ar_text_embedding
        self.ar_text_position = raw_t2s.model.ar_text_position
        self.ar_audio_embedding = raw_t2s.model.ar_audio_embedding
        self.ar_audio_position = raw_t2s.model.ar_audio_position
        
        # self.t2s_transformer = T2STransformer(self.num_layers, blocks)
        self.t2s_transformer = raw_t2s.model.t2s_transformer

        # self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.ar_predict_layer = raw_t2s.model.ar_predict_layer
        # self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
        self.max_sec = self.config["data"]["max_sec"]
        self.top_k = int(self.config["inference"]["top_k"])
        self.early_stop_num = torch.LongTensor([self.hz * self.max_sec])
    
    # def forward(self, x:LongTensor, prompts:LongTensor):
    def forward(self,prompts:LongTensor, ref_seq:LongTensor, text_seq:LongTensor, ref_bert:torch.Tensor, text_bert:torch.Tensor):
        bert = torch.cat([ref_bert.T, text_bert.T], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)

        x = self.ar_text_embedding(all_phoneme_ids)
        x = x + self.bert_proj(bert.transpose(1, 2))
        x:torch.Tensor = self.ar_text_position(x)

        early_stop_num = self.early_stop_num


        #[1,N,512] [1,N]
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
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)\
                                                .unsqueeze(0)\
                                                .expand(bsz*self.num_head, -1, -1)\
                                                .view(bsz, self.num_head, src_len, src_len)\
                                                .to(device=x.device, dtype=torch.bool)
        
        idx = 0
        
        xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)

        logits = self.ar_predict_layer(xy_dec[:, -1])
        logits = logits[:, :-1]
        samples = sample(logits, y, top_k=self.top_k, top_p=1, repetition_penalty=1.35, temperature=1.0)[0]
        y = torch.concat([y, samples], dim=1)
        y_emb = self.ar_audio_embedding(y[:, -1:])
        xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to(dtype=y_emb.dtype,device=y_emb.device)

        stop = False
        # for idx in range(1, 50):
        for idx in range(1, 1500):
            #[1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
            # y, k, v, y_emb, logits, samples = self.stage_decoder(y, k, v, y_emb, x_example)
            xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache)
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = sample(logits, y, top_k=self.top_k, top_p=1, repetition_penalty=1.35, temperature=1.0)[0]

            y = torch.concat([y, samples], dim=1)
            
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                break

            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to(dtype=y_emb.dtype,device=y_emb.device)
        
        y[0, -1] = 0

        return y[:, -idx:].unsqueeze(0)
    
    # def first_stage_decoder(self, x, prompt):
    #     y = prompt
    #     x_example = x[:,:,0] * 0.0
    #     #N, 1, 512
    #     cache = {
    #         "all_stage": self.num_layers,
    #         "k": None,
    #         "v": None,
    #         "y_emb": None,
    #         "first_infer": 1,
    #         "stage": 0,
    #     }

    #     y_emb = self.ar_audio_embedding(y)

    #     cache["y_emb"] = y_emb
    #     y_pos = self.ar_audio_position(y_emb)

    #     xy_pos = torch.concat([x, y_pos], dim=1)

    #     y_example = y_pos[:,:,0] * 0.0
    #     x_attn_mask = torch.matmul(x_example.transpose(0, 1) , x_example).bool()
    #     y_attn_mask = torch.ones_like(torch.matmul(y_example.transpose(0, 1), y_example), dtype=torch.int64)
    #     y_attn_mask = torch.cumsum(y_attn_mask, dim=1) - torch.cumsum(
    #         torch.ones_like(y_example.transpose(0, 1), dtype=torch.int64), dim=0
    #     )
    #     y_attn_mask = y_attn_mask > 0

    #     x_y_pad = torch.matmul(x_example.transpose(0, 1), y_example).bool()
    #     y_x_pad = torch.matmul(y_example.transpose(0, 1), x_example).bool()
    #     x_attn_mask_pad = torch.cat([x_attn_mask, torch.ones_like(x_y_pad)], dim=1)
    #     y_attn_mask = torch.cat([y_x_pad, y_attn_mask], dim=1)
    #     xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
    #     cache["k"] = torch.matmul(x_attn_mask_pad[0].float().unsqueeze(-1), torch.zeros((1, 512)))\
    #     .unsqueeze(1).repeat(self.num_layers, 1, 1, 1)
    #     cache["v"] = torch.matmul(x_attn_mask_pad[0].float().unsqueeze(-1), torch.zeros((1, 512)))\
    #     .unsqueeze(1).repeat(self.num_layers, 1, 1, 1)

    #     xy_dec = self.h(xy_pos, mask=xy_attn_mask, cache=cache)
    #     logits = self.ar_predict_layer(xy_dec[:, -1])
    #     samples = sample(logits[0], y, top_k=self.top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)

    #     y = torch.concat([y, samples], dim=1)

    #     return y, cache["k"], cache["v"], cache["y_emb"], x_example
    
    # def stage_decoder(self, y, k, v, y_emb, x_example):
    #     cache = {
    #         "all_stage": self.num_layers,
    #         "k": torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 1)),
    #         "v": torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 1)),
    #         "y_emb": y_emb,
    #         "first_infer": 0,
    #         "stage": 0,
    #     }

    #     y_emb = torch.cat(
    #         [cache["y_emb"], self.ar_audio_embedding(y[:, -1:])], 1
    #     )
    #     cache["y_emb"] = y_emb
    #     y_pos = self.ar_audio_position(y_emb)

    #     xy_pos = y_pos[:, -1:]
        
    #     y_example = y_pos[:,:,0] * 0.0

    #     xy_attn_mask = torch.cat([x_example, y_example], dim=1)
    #     xy_attn_mask = torch.zeros_like(xy_attn_mask, dtype=torch.bool)

    #     xy_dec = self.h(xy_pos, mask=xy_attn_mask, cache=cache)
    #     logits = self.ar_predict_layer(xy_dec[:, -1])
    #     samples = sample(logits[0], y, top_k=self.top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)

    #     y = torch.concat([y, samples], dim=1)

    #     return y, cache["k"], cache["v"], cache["y_emb"], logits, samples
    


bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
cnhubert.cnhubert_base_path = cnhubert_base_path

class BertModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(BertModel, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask, token_type_ids, word2ph:IntTensor):
        res = self.bert(input_ids, attention_mask, token_type_ids)
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

    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, token_type_ids:torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        res = torch.cat(outputs["hidden_states"][-3:-2], -1)[0][1:-1]
        return res

class SSLModel(torch.nn.Module):
    def __init__(self,vits:VitsModel):
        super().__init__()
        self.ssl = cnhubert.get_model().model
        self.ssl_proj = vits.vq_model.ssl_proj
        self.quantizer = vits.vq_model.quantizer

    def forward(self, ref_audio)->LongTensor:
        ref_audio_16k,ref_audio_sr = parse_audio(ref_audio)
        ssl_content = self.ssl(ref_audio_16k)["last_hidden_state"].transpose(1, 2)
        codes = self.extract_latent(ssl_content.float())
        prompt_semantic = codes[0, 0]
        prompts = prompt_semantic.unsqueeze(0)
        return prompts,ref_audio_sr

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)

def export_bert(tokenizer):
    ref_bert_inputs = tokenizer("在参加挼死特春晚的时候有人问了这样一个问题", return_tensors="pt")
    ref_bert_inputs = {
        'input_ids': torch.jit.annotate(torch.Tensor,ref_bert_inputs['input_ids']),
        'attention_mask': torch.jit.annotate(torch.Tensor,ref_bert_inputs['attention_mask']),
        'token_type_ids': torch.jit.annotate(torch.Tensor,ref_bert_inputs['token_type_ids']),
    }
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path,output_hidden_states=True)
    my_bert_model = MyBertModel(bert_model)

    my_bert_model = torch.jit.trace(my_bert_model,example_kwarg_inputs=ref_bert_inputs)
    print('trace my_bert_model')
    bert = BertModel(my_bert_model)
    torch.jit.script(bert).save("onnx/bert_model.pt")
    print('exported bert')
    
def export(gpt_path, vits_path):
    # gpt_path = "GPT_weights_v2/xw-e15.ckpt"
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    raw_t2s = get_raw_t2s_model(dict_s1)
    t2s_m = T2SModel(dict_s1['config'],raw_t2s,top_k=3)
    t2s_m.eval()
    torch.jit.script(t2s_m).save("onnx/xw/t2s_model.pt")
    print('exported t2s_m')

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    
    ref_bert_inputs = tokenizer("声音,是有温度的.夜晚的声音,会发光", return_tensors="pt")
    ref_bert_inputs['word2ph'] = torch.Tensor([2,2,1,2,2,2,2,2,1,2,2,2,2,2,1,2,2,2]).int()

    text_berf_inputs = tokenizer("大家好,我有一个春晚问题.", return_tensors="pt")
    text_berf_inputs['word2ph'] = torch.Tensor([2,2,2,1,2,2,2,2,2,2,2,2,1]).int()

    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path,output_hidden_states=True)

    my_bert_model = MyBertModel(bert_model)

    bert = BertModel(my_bert_model)

    # export_bert(tokenizer)


    # vits_path = "SoVITS_weights_v2/xw_e8_s216.pth"
    vits = VitsModel(vits_path)
    vits.eval()
    
    ref_audio = torch.tensor([load_audio("output/denoise_opt/xw.mp3_0000000000_0000156480.wav", 48000)]).float()
    ssl = SSLModel(vits)
    torch.jit.trace(ssl,example_inputs=(torch.jit.annotate(torch.Tensor,ref_audio))).save("onnx/xw/ssl_model.pt")
    print('exported ssl')

    # ref_seq = torch.LongTensor([cleaned_text_to_sequence(["zh", "ai4", "ch", "an1","j" ,"ia1","r","ua4","s","i3","t","e3","ch","un1","w","an3","d","e1", "sh", "i2", "h", "ou4", "y", "ou3", "r", "en2","w","en4","l","e1","zh","e4","y","ang4","y","i2","g","e4","w","en4","t","i2"],version='v2')])
    ref_seq = torch.LongTensor([cleaned_text_to_sequence(['sh','eng1','y','in1',',','sh','i4','y','ou3','w','en1','d','u4','d','e','.','y','e4','w','an3','d','e','sh','eng1','y','in1',',','h','ui4','f','a1','g','uang1'],version='v2')])
    text_seq = torch.LongTensor([cleaned_text_to_sequence(["d", "a4", "j", "ia1", "h", "ao3",",","w","o3","y", "ou3","y","i2","g","e4","q","i2","g","uai4","w","en4","t","i2","."],version='v2')])
    ref_bert = bert(**ref_bert_inputs)
    text_bert = bert(**text_berf_inputs)
    prompts,ref_audio_sr = ssl(ref_audio)
    pred_semantic = t2s_m(prompts, ref_seq, text_seq, ref_bert, text_bert)
    
    torch.jit.trace(vits,example_inputs=(
        torch.jit.annotate(torch.Tensor,text_seq),
        torch.jit.annotate(torch.Tensor,pred_semantic),
        torch.jit.annotate(torch.Tensor,ref_audio_sr))).save("onnx/xw/vits_model.pt")
    print('exported vits')

@torch.jit.script
def parse_audio(ref_audio):
    ref_audio_16k = torchaudio.functional.resample(ref_audio,48000,16000).float()#.to(ref_audio.device)
    ref_audio_sr = torchaudio.functional.resample(ref_audio,48000,32000).float()#.to(ref_audio.device)
    return ref_audio_16k,ref_audio_sr

def test():

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    # bert_model = AutoModelForMaskedLM.from_pretrained(bert_path,output_hidden_states=True)
    # bert_model.bert.embeddings = MyBertEmbeddings(bert_model.bert.config)
    # bert_model.bert.encoder = MyBertEncoder(bert_model.bert.config)
    # my_bert_model = MyBertModel(bert_model)
    # bert = BertModel(my_bert_model)
    bert = torch.jit.load("onnx/bert_model.pt",map_location='cuda')

    # gpt_path = "GPT_weights_v2/xw-e15.ckpt"
    # dict_s1 = torch.load(gpt_path, map_location="cpu")
    # raw_t2s = get_raw_t2s_model(dict_s1)
    # t2s = T2SModel(dict_s1['config'],raw_t2s,top_k=3)
    # t2s.eval()
    t2s = torch.jit.load("onnx/xw/t2s_model.pt",map_location='cuda')

    # vits_path = "SoVITS_weights_v2/xw_e8_s216.pth"
    # vits = VitsModel(vits_path).to('cuda')
    # vits.eval()

    # ssl = SSLModel(vits).to('cuda')
    # ssl.eval()

    vits = torch.jit.load("onnx/xw/vits_model.pt",map_location='cuda')

    ssl = torch.jit.load("onnx/xw/ssl_model.pt",map_location='cuda')

    ref_seq = torch.LongTensor([cleaned_text_to_sequence(["zh", "ai4", "ch", "an1","j" ,"ia1","r","ua4","s","i3","t","e3","ch","un1","w","an3","d","e1", "sh", "i2", "h", "ou4", "y", "ou3", "r", "en2","w","en4","l","e1","zh","e4","y","ang4","y","i2","g","e4","w","en4","t","i2"],version='v2')])
    ref_seq=ref_seq.to('cuda')
    text_seq = torch.LongTensor([cleaned_text_to_sequence(["d", "a4", "j", "ia1", "h", "ao3",",","w","o3","y", "ou3","y","i2","g","e4","q","i2","g","uai4","d","e1","w","en4","t","i2","."],version='v2')])
    text_seq=text_seq.to('cuda')

    ref_bert_inputs = tokenizer("在参加挼死特春晚的时候有人问了这样一个问题", return_tensors="pt")
    ref_bert_inputs['word2ph'] = torch.Tensor([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]).int().to('cuda')
    ref_bert = bert(
        ref_bert_inputs['input_ids'].to('cuda'), 
        ref_bert_inputs['attention_mask'].to('cuda'), 
        ref_bert_inputs['token_type_ids'].to('cuda'), 
        ref_bert_inputs['word2ph'].to('cuda'))
    
    print('ref_bert:',ref_bert.device)

    text_berf_inputs = tokenizer("大家好,我有一个奇怪的问题.", return_tensors="pt")
    text_berf_inputs['word2ph'] = torch.Tensor([2,2,2,1,2,2,2,2,2,2,2,2,2,1]).int().to('cuda')
    text_bert = bert(text_berf_inputs['input_ids'].to('cuda'), 
                     text_berf_inputs['attention_mask'].to('cuda'), 
                     text_berf_inputs['token_type_ids'].to('cuda'), 
                     text_berf_inputs['word2ph'])

    ref_audio = torch.tensor([load_audio("output/denoise_opt/xw.mp3_0000000000_0000156480.wav", 48000)]).float().to('cuda')
    
    print('start ssl')
    prompts,ref_audio_sr = ssl(ref_audio)

    pred_semantic = t2s(prompts, ref_seq, text_seq, ref_bert, text_bert)

    print('start vits:',pred_semantic.shape)
    print('ref_audio_sr:',ref_audio_sr.device)
    audio = vits(text_seq, pred_semantic, ref_audio_sr)
    print('start write wav')
    soundfile.write("out.wav", audio.detach().cpu().numpy(), 32000)
    torch.load("onnx/symbols_v2.json")

    # audio = vits(text_seq, pred_semantic1, ref_audio)
    # soundfile.write("out.wav", audio, 32000)

import text
import json

def export_symbel(version='v2'):
    if version=='v1':
        symbols = text._symbol_to_id_v1
        with open(f"onnx/symbols_v1.json", "w") as file:
            json.dump(symbols, file, indent=4)
    else:
        symbols = text._symbol_to_id_v2
        with open(f"onnx/symbols_v2.json", "w") as file:
            json.dump(symbols, file, indent=4)

if __name__ == "__main__":
    export(gpt_path="GPT_weights_v2/chen1-e15.ckpt", vits_path="SoVITS_weights_v2/chen1_e8_s208.pth")
    # test()
    # export_symbel()
    # tokenizer = AutoTokenizer.from_pretrained(bert_path)
    # text_berf_inputs = tokenizer("大家好,我有一个奇怪的问题.", return_tensors="pt")
    # print(text_berf_inputs)
    # ref_audio = load_audio("output/denoise_opt/chen1.mp4_0000033600_0000192000.wav", 48000)
    # print(ref_audio.shape)
    # soundfile.write("chen1_ref.wav", ref_audio, 48000)