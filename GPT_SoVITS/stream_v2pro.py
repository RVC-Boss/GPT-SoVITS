# 这是一个实验性质的实现，旨在探索 stream infer 的可能性。(xiao hai xie zhe wan de)
from typing import List
from export_torch_script import ExportERes2NetV2, SSLModel, T2SModel, VitsModel, get_raw_t2s_model, init_sv_cn, resamplex, sample, spectrogram_torch
import export_torch_script
from my_utils import load_audio
import torch
from torch import LongTensor, Tensor, nn
from torch.nn import functional as F

import soundfile
from inference_webui import get_phones_and_bert
import matplotlib.pyplot as plt


class StreamT2SModel(nn.Module):
    def __init__(self, t2s: T2SModel):
        super(StreamT2SModel, self).__init__()
        self.t2s = t2s

    @torch.jit.export
    def pre_infer(
        self,
        prompts: LongTensor,
        ref_seq: LongTensor,
        text_seq: LongTensor,
        ref_bert: torch.Tensor,
        text_bert: torch.Tensor,
        top_k: int,
    ) -> tuple[int, Tensor, Tensor, List[Tensor], List[Tensor]]:
        bert = torch.cat([ref_bert.T, text_bert.T], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)

        x = self.t2s.ar_text_embedding(all_phoneme_ids)
        x = x + self.t2s.bert_proj(bert.transpose(1, 2))
        x: torch.Tensor = self.t2s.ar_text_position(x)

        # [1,N,512] [1,N]
        # y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)
        y = prompts
        # x_example = x[:,:,0] * 0.0

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        y_emb = self.t2s.ar_audio_embedding(y)
        y_len: int = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_pos = self.t2s.ar_audio_position(y_emb)
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
            .expand(bsz * self.t2s.num_head, -1, -1)
            .view(bsz, self.t2s.num_head, src_len, src_len)
            .to(device=x.device, dtype=torch.bool)
        )

        xy_dec, k_cache, v_cache = self.t2s.t2s_transformer.process_prompt(
            xy_pos, xy_attn_mask, None
        )

        logits = self.t2s.ar_predict_layer(xy_dec[:, -1])
        logits = logits[:, :-1]
        samples = sample(
            logits, y, top_k=top_k, top_p=1, repetition_penalty=1.35, temperature=1.0
        )[0]
        y = torch.concat([y, samples], dim=1)
        y_emb: Tensor = self.t2s.ar_audio_embedding(y[:, -1:])
        xy_pos: Tensor = (
            y_emb * self.t2s.ar_audio_position.x_scale
            + self.t2s.ar_audio_position.alpha
            * self.t2s.ar_audio_position.pe[:, y_len].to(
                dtype=y_emb.dtype, device=y_emb.device
            )
        )

        return y_len, y, xy_pos, k_cache, v_cache

    @torch.jit.export
    def decode_next_token(
        self,
        idx: int,  # 记住从1开始 到1500
        top_k: int,
        y_len: int,
        y: Tensor,
        xy_pos: Tensor,
        k_cache: List[Tensor],
        v_cache: List[Tensor],
    ) -> tuple[Tensor, Tensor, int, List[Tensor], List[Tensor]]:
        # [1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
        # y, k, v, y_emb, logits, samples = self.stage_decoder(y, k, v, y_emb, x_example)
        xy_dec, k_cache, v_cache = self.t2s.t2s_transformer.decode_next_token(
            xy_pos, k_cache, v_cache
        )
        logits = self.t2s.ar_predict_layer(xy_dec[:, -1])

        if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
            logits = logits[:, :-1]

        samples = sample(
            logits, y, top_k=top_k, top_p=1, repetition_penalty=1.35, temperature=1.0
        )[0]

        y = torch.concat([y, samples], dim=1)
        last_token = int(samples[0, 0])

        # if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
        #     stop = True
        if torch.argmax(logits, dim=-1)[0] == self.t2s.EOS or samples[0, 0] == self.t2s.EOS:
            return y[:,:-1], xy_pos, self.t2s.EOS, k_cache, v_cache

        # if stop:
        #     if y.shape[1] == 0:
        #         y = torch.concat([y, torch.zeros_like(samples)], dim=1)
        #     break

        y_emb = self.t2s.ar_audio_embedding(y[:, -1:])
        xy_pos = (
            y_emb * self.t2s.ar_audio_position.x_scale
            + self.t2s.ar_audio_position.alpha
            * self.t2s.ar_audio_position.pe[:, y_len + idx].to(
                dtype=y_emb.dtype, device=y_emb.device
            )
        )
        return y, xy_pos, last_token, k_cache, v_cache

    def forward(
        self,
        idx: int,  # 记住从1开始 到1500
        top_k: int,
        y_len: int,
        y: Tensor,
        xy_pos: Tensor,
        k_cache: List[Tensor],
        v_cache: List[Tensor],
    ):
        return self.decode_next_token(idx,top_k,y_len,y,xy_pos,k_cache,v_cache)


class StepVitsModel(nn.Module):
    def __init__(self, vits: VitsModel,sv_model:ExportERes2NetV2):
        super().__init__()
        self.hps = vits.hps
        self.vq_model = vits.vq_model
        self.hann_window = vits.hann_window
        self.sv = sv_model

    def ref_handle(self, ref_audio_32k):
        refer = spectrogram_torch(
            self.hann_window,
            ref_audio_32k.float(),
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        refer = refer.to(ref_audio_32k.dtype)
        ref_audio_16k = resamplex(ref_audio_32k, 32000, 16000).to(ref_audio_32k.dtype).to(ref_audio_32k.device)
        sv_emb = self.sv(ref_audio_16k)
        return refer, sv_emb

    def extract_latent(self, ssl_content):
        codes = self.vq_model.extract_latent(ssl_content)
        return codes[0]

    def forward(self, pred_semantic, text_seq, refer, sv_emb=None):
        return self.vq_model(
            pred_semantic, text_seq, refer, speed=1.0, sv_emb=sv_emb
        )[0, 0]


@torch.jit.script
def find_best_audio_offset_fast(reference_audio: Tensor, search_audio: Tensor):
    ref_len = len(reference_audio)
    search_len = len(search_audio)

    if search_len < ref_len:
        raise ValueError(
            f"搜索音频长度 ({search_len}) 必须大于等于参考音频长度 ({ref_len})"
        )

    # 使用F.conv1d计算原始互相关
    reference_flipped = reference_audio.unsqueeze(0).unsqueeze(0)
    search_padded = search_audio.unsqueeze(0).unsqueeze(0)

    # 计算点积
    dot_products = F.conv1d(search_padded, reference_flipped).squeeze()

    if len(dot_products.shape) == 0:
        dot_products = dot_products.unsqueeze(0)

    # 计算参考音频的平方和
    ref_squared_sum = torch.sum(reference_audio**2)

    # 计算搜索音频每个位置的平方和（滑动窗口）
    search_squared = search_audio**2
    search_squared_padded = search_squared.unsqueeze(0).unsqueeze(0)
    ones_kernel = torch.ones(
        1, 1, ref_len, dtype=search_audio.dtype, device=search_audio.device
    )

    segment_squared_sums = F.conv1d(search_squared_padded, ones_kernel).squeeze()

    if len(segment_squared_sums.shape) == 0:
        segment_squared_sums = segment_squared_sums.unsqueeze(0)

    # 计算归一化因子
    ref_norm = torch.sqrt(ref_squared_sum)
    segment_norms = torch.sqrt(segment_squared_sums)

    # 避免除零
    epsilon = 1e-8
    normalization_factor = ref_norm * segment_norms + epsilon

    # 归一化互相关
    correlation_scores = dot_products / normalization_factor

    best_offset = torch.argmax(correlation_scores).item()

    return best_offset, correlation_scores


import time

def test_stream(
    gpt_path,
    vits_path,
    version,
    ref_audio_path,
    ref_text,
    output_path,
    device="cpu",
    is_half=True,
):
    if export_torch_script.sv_cn_model == None:
        init_sv_cn(device,is_half)

    ref_audio = torch.tensor([load_audio(ref_audio_path, 16000)]).float()
    ssl = SSLModel()

    print(f"device: {device}")

    ref_seq_id, ref_bert_T, ref_norm_text = get_phones_and_bert(
        ref_text, "all_zh", "v2"
    )
    ref_seq = torch.LongTensor([ref_seq_id]).to(device)
    ref_bert = ref_bert_T.T
    if is_half:
        ref_bert = ref_bert.half()
    ref_bert = ref_bert.to(ref_seq.device)

    text_seq_id, text_bert_T, norm_text = get_phones_and_bert(
        "这是一个简单的示例，真没想到这么简单就完成了，真的神奇，接下来我们说说狐狸,可能这就是狐狸吧.它有长长的尾巴，尖尖的耳朵，传说中还有九条尾巴。你觉得狐狸神奇吗？", "auto", "v2"
    )
    text_seq = torch.LongTensor([text_seq_id]).to(device)
    text_bert = text_bert_T.T
    if is_half:
        text_bert = text_bert.half()
    text_bert = text_bert.to(text_seq.device)

    ssl_content = ssl(ref_audio)
    if is_half:
        ssl_content = ssl_content.half()
    ssl_content = ssl_content.to(device)

    sv_model = ExportERes2NetV2(export_torch_script.sv_cn_model)

    # vits_path = "SoVITS_weights_v2/xw_e8_s216.pth"
    vits = VitsModel(vits_path, version,is_half=is_half,device=device)
    vits.eval()

    # gpt_path = "GPT_weights_v2/xw-e15.ckpt"
    # dict_s1 = torch.load(gpt_path, map_location=device)
    dict_s1 = torch.load(gpt_path, weights_only=False)
    raw_t2s = get_raw_t2s_model(dict_s1).to(device)
    print("#### get_raw_t2s_model ####")
    print(raw_t2s.config)
    if is_half:
        raw_t2s = raw_t2s.half()
    t2s_m = T2SModel(raw_t2s)
    t2s_m.eval()
    # t2s = torch.jit.script(t2s_m).to(device)
    t2s = t2s_m
    print("#### script t2s_m ####")

    print("vits.hps.data.sampling_rate:", vits.hps.data.sampling_rate)

    stream_t2s = StreamT2SModel(t2s).to(device)
    stream_t2s = torch.jit.script(stream_t2s)

    ref_audio_sr = resamplex(ref_audio, 16000, 32000)
    if is_half:
        ref_audio_sr = ref_audio_sr.half()
    ref_audio_sr = ref_audio_sr.to(device)

    top_k = 15

    codes = vits.vq_model.extract_latent(ssl_content)
    prompt_semantic = codes[0, 0]
    prompts = prompt_semantic.unsqueeze(0)

    audio_16k = resamplex(ref_audio_sr, 32000, 16000).to(ref_audio_sr.dtype)
    sv_emb = sv_model(audio_16k)
    print("text_seq",text_seq.shape)

    refer = spectrogram_torch(
        vits.hann_window,
        ref_audio_sr,
        vits.hps.data.filter_length,
        vits.hps.data.sampling_rate,
        vits.hps.data.hop_length,
        vits.hps.data.win_length,
        center=False,
    )

    st = time.time()
    et = time.time()

    y_len, y, xy_pos, k_cache, v_cache = stream_t2s.pre_infer(prompts, ref_seq, text_seq, ref_bert, text_bert, top_k)
    idx = 1
    last_idx = 0
    audios = []
    raw_audios = []
    last_audio_ret = None
    offset_index = []
    full_audios = []
    print("y.shape:", y.shape)
    cut_id = 0
    while True:
        y, xy_pos, last_token, k_cache, v_cache = stream_t2s(idx, top_k, y_len, y, xy_pos, k_cache, v_cache)
        # print("y.shape:", y.shape)
        stop = last_token==t2s.EOS
        print('idx:',idx , 'y.shape:', y.shape, y.shape[1]-idx)

        if last_token < 50 and idx-last_idx > (len(audios)+1) * 25 and idx > cut_id:
            cut_id = idx + 7
            print('trigger:',idx, last_idx, y[:,-idx+last_idx:], y[:,-idx+last_idx:].shape)
            # y = torch.cat([y, y[:,-1:]], dim=1)
            # idx+=1

        if stop :
            idx -=1
            print('stop')
            print(idx, y[:,-idx+last_idx:])
            print(idx,last_idx, y.shape)
            print(y[:,-idx:-idx+20])


        # 玄学这档子事说不清楚
        if idx == cut_id or stop:
            print(f"idx: {idx}, last_idx: {last_idx}, cut_id: {cut_id}, stop: {stop}")
            audio = vits.vq_model(y[:,-idx:].unsqueeze(0), text_seq, refer, speed=1.0, sv_emb=sv_emb)[0, 0]
            full_audios.append(audio)
            if last_idx == 0:
                last_audio_ret = audio[-1280*8:-1280*8+256]
                audio = audio[:-1280*8]
                raw_audios.append(audio)
                et = time.time()
            else:
                if stop:
                    audio_ = audio[last_idx*1280 -1280*8:]
                    raw_audios.append(audio_)
                    i, x = find_best_audio_offset_fast(last_audio_ret, audio_[:1280])
                    offset_index.append(i)
                    audio = audio_[i:]
                else:
                    audio_ = audio[last_idx*1280 -1280*8:-1280*8]
                    raw_audios.append(audio_)
                    i, x = find_best_audio_offset_fast(last_audio_ret, audio_[:1280])
                    offset_index.append(i)
                    last_audio_ret = audio[-1280*8:-1280*8+256]
                    audio = audio_[i:]
            last_idx = idx
            # print(f'write {output_path}/out_{audio_index}')
            # soundfile.write(f"{output_path}/out_{audio_index}.wav", audio.float().detach().cpu().numpy(), 32000)
            audios.append(audio)
        # print(idx,'/',1500 , y.shape, y[0,-1].item(), stop)
        if idx>1500:
            break

        if stop:
            break

        idx+=1

    at = time.time()

    for (i,a) in enumerate(audios):
        print(f'write {output_path}/out_{i}')
        soundfile.write(f"{output_path}/out_{i}.wav", a.float().detach().cpu().numpy(), 32000)
        
    print(f"frist token: {et - st:.4f} seconds")
    print(f"all token: {at - st:.4f} seconds")
    audio = vits.vq_model(y[:,-idx:].unsqueeze(0), text_seq, refer, speed=1.0, sv_emb=sv_emb)[0, 0]
    soundfile.write(f"{output_path}/out_final.wav", audio.float().detach().cpu().numpy(), 32000)
    audio = torch.cat(audios, dim=0)
    soundfile.write(f"{output_path}/out.wav", audio.float().detach().cpu().numpy(), 32000)
    audio_raw = torch.cat(raw_audios, dim=0)
    soundfile.write(f"{output_path}/out.raw.wav", audio_raw.float().detach().cpu().numpy(), 32000)
    

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    max_duration = full_audios[-1].shape[0]
    plt.xlim(0, max_duration)

    last_line = 0

    for i,a in enumerate(full_audios):
        plt.plot((a+2.0*i).float().detach().cpu().numpy(), color=colors[i], alpha=0.5, label=f"Audio {i}")
        # plt.axvline(x=last_line, color=colors[i], linestyle='--')
        last_line = a.shape[0]-8*1280
        plt.axvline(x=last_line, color=colors[i], linestyle='--')
    
    plt.plot((audio-2.0).float().detach().cpu().numpy(), color='black', label='Final Audio')

    plt.plot((audio_raw-4.0).float().detach().cpu().numpy(), color='cyan', label='Raw Audio')

    print("offset_index:", offset_index)
    plt.show()


def export_prov2(
    gpt_path,
    vits_path,
    version,
    ref_audio_path,
    ref_text,
    output_path,
    device="cpu",
    is_half=True,
    lang="auto",
):
    if export_torch_script.sv_cn_model == None:
        init_sv_cn(device,is_half)

    ref_audio = torch.tensor([load_audio(ref_audio_path, 16000)]).float()
    ssl = SSLModel()

    print(f"device: {device}")

    ref_seq_id, ref_bert_T, ref_norm_text = get_phones_and_bert(
        ref_text, lang, "v2"
    )
    ref_seq = torch.LongTensor([ref_seq_id]).to(device)
    ref_bert = ref_bert_T.T
    if is_half:
        ref_bert = ref_bert.half()
    ref_bert = ref_bert.to(ref_seq.device)

    text_seq_id, text_bert_T, norm_text = get_phones_and_bert(
        "这是一个简单的示例，真没想到这么简单就完成了.The King and His Stories.Once there was a king.He likes to write stories, but his stories were not good.", "auto", "v2"
    )
    text_seq = torch.LongTensor([text_seq_id]).to(device)
    text_bert = text_bert_T.T
    if is_half:
        text_bert = text_bert.half()
    text_bert = text_bert.to(text_seq.device)

    ssl_content = ssl(ref_audio)
    if is_half:
        ssl_content = ssl_content.half()
    ssl_content = ssl_content.to(device)

    sv_model = ExportERes2NetV2(export_torch_script.sv_cn_model)

    # vits_path = "SoVITS_weights_v2/xw_e8_s216.pth"
    vits = VitsModel(vits_path, version,is_half=is_half,device=device)
    vits.eval()
    vits = StepVitsModel(vits, sv_model)

    # gpt_path = "GPT_weights_v2/xw-e15.ckpt"
    # dict_s1 = torch.load(gpt_path, map_location=device)
    dict_s1 = torch.load(gpt_path, weights_only=False)
    raw_t2s = get_raw_t2s_model(dict_s1).to(device)
    print("#### get_raw_t2s_model ####")
    print(raw_t2s.config)
    if is_half:
        raw_t2s = raw_t2s.half()
    t2s_m = T2SModel(raw_t2s)
    t2s_m.eval()
    # t2s = torch.jit.script(t2s_m).to(device)
    t2s = t2s_m
    print("#### script t2s_m ####")

    print("vits.hps.data.sampling_rate:", vits.hps.data.sampling_rate)

    stream_t2s = StreamT2SModel(t2s).to(device)
    stream_t2s = torch.jit.script(stream_t2s)

    ref_audio_sr = resamplex(ref_audio, 16000, 32000)
    ref_audio_sr = ref_audio_sr.to(device)
    if is_half:
        ref_audio_sr = ref_audio_sr.half()

    top_k = 15

    prompts = vits.extract_latent(ssl_content)

    audio_16k = resamplex(ref_audio_sr, 32000, 16000).to(ref_audio_sr.dtype)
    sv_emb = sv_model(audio_16k)
    print("text_seq",text_seq.shape)
    # torch.jit.trace()

    refer,sv_emb = vits.ref_handle(ref_audio_sr)

    st = time.time()
    et = time.time()

    y_len, y, xy_pos, k_cache, v_cache = stream_t2s.pre_infer(prompts, ref_seq, text_seq, ref_bert, text_bert, top_k)
    idx = 1
    print("y.shape:", y.shape)
    while True:
        y, xy_pos, last_token, k_cache, v_cache = stream_t2s(idx, top_k, y_len, y, xy_pos, k_cache, v_cache)
        # print("y.shape:", y.shape)

        idx+=1
        # print(idx,'/',1500 , y.shape, y[0,-1].item(), stop)
        if idx>1500:
            break

        if last_token == t2s.EOS:
            break

    at = time.time()
    print("EOS:",t2s.EOS)

    print(f"frist token: {et - st:.4f} seconds")
    print(f"all token: {at - st:.4f} seconds")
    print("sv_emb", sv_emb.shape)
    print("refer",refer.shape)
    y = y[:,-idx:].unsqueeze(0)
    print("y", y.shape)
    audio = vits(y, text_seq, refer, sv_emb)
    soundfile.write(f"{output_path}/out_final.wav", audio.float().detach().cpu().numpy(), 32000)

    torch._dynamo.mark_dynamic(ssl_content, 2)
    torch._dynamo.mark_dynamic(ref_audio_sr, 1)
    torch._dynamo.mark_dynamic(ref_seq, 1)
    torch._dynamo.mark_dynamic(text_seq, 1)
    torch._dynamo.mark_dynamic(ref_bert, 0)
    torch._dynamo.mark_dynamic(text_bert, 0)
    torch._dynamo.mark_dynamic(refer, 2)
    torch._dynamo.mark_dynamic(y, 2)

    inputs = {
        "forward": (y, text_seq, refer, sv_emb),
        "extract_latent": ssl_content,
        "ref_handle": ref_audio_sr,
    }

    stream_t2s.save(f"{output_path}/t2s.pt")
    torch.jit.trace_module(vits, inputs=inputs, optimize=True).save(f"{output_path}/vits.pt")
    torch.jit.script(find_best_audio_offset_fast, optimize=True).save(f"{output_path}/find_best_audio_offset_fast.pt")

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument("--gpt_model", required=True, help="Path to the GPT model file")
    parser.add_argument(
        "--sovits_model", required=True, help="Path to the SoVITS model file"
    )
    parser.add_argument(
        "--ref_audio", required=True, help="Path to the reference audio file"
    )
    parser.add_argument(
        "--ref_text", required=True, help="Path to the reference text file"
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to the output directory"
    )
    parser.add_argument("--device", help="Device to use", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--version", help="version of the model", default="v2Pro")
    parser.add_argument("--no-half", action="store_true", help = "Do not use half precision for model weights")
    parser.add_argument("--lang", default="auto", help="Language for text processing (default: auto)")

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    is_half = not args.no_half
    with torch.no_grad():
        export_prov2(
            gpt_path=args.gpt_model,
            vits_path=args.sovits_model,
            version=args.version,
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
            output_path=args.output_path,
            device=args.device,
            is_half=is_half,
            lang=args.lang,
        )
