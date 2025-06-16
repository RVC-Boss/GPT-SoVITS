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


class StreamT2SModel(nn.Module):
    def __init__(self, t2s: T2SModel):
        super(StreamT2SModel, self).__init__()
        self.t2s = t2s
        self.k_cache: list[torch.Tensor] = [torch.zeros([1])]
        self.v_cache: list[torch.Tensor] = [torch.zeros([1])]

    @torch.jit.export
    def pre_infer(
        self,
        prompts: LongTensor,
        ref_seq: LongTensor,
        text_seq: LongTensor,
        ref_bert: torch.Tensor,
        text_bert: torch.Tensor,
        top_k: int,
    ) -> tuple[int, Tensor, Tensor]:
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

        self.k_cache = k_cache
        self.v_cache = v_cache
        return y_len, y, xy_pos

    @torch.jit.export
    def decode_next_token(
        self,
        idx: int,  # 记住从1开始 到1500
        top_k: int,
        y_len: int,
        y: Tensor,
        xy_pos: Tensor,
    ) -> tuple[Tensor, Tensor, bool]:
        # [1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
        # y, k, v, y_emb, logits, samples = self.stage_decoder(y, k, v, y_emb, x_example)
        xy_dec, k_cache, v_cache = self.t2s.t2s_transformer.decode_next_token(
            xy_pos, self.k_cache, self.v_cache
        )
        logits = self.t2s.ar_predict_layer(xy_dec[:, -1])

        if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
            logits = logits[:, :-1]

        samples = sample(
            logits, y, top_k=top_k, top_p=1, repetition_penalty=1.35, temperature=1.0
        )[0]

        y = torch.concat([y, samples], dim=1)

        # if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
        #     stop = True
        if torch.argmax(logits, dim=-1)[0] == self.t2s.EOS or samples[0, 0] == self.t2s.EOS:
            self.k_cache = [torch.zeros([1])]
            self.v_cache = [torch.zeros([1])]
            return y[:,:-1], xy_pos, True

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
        return y, xy_pos, False

    def forward(
        self,
        idx: int,  # 记住从1开始 到1500
        top_k: int,
        y_len: int,
        y: Tensor,
        xy_pos: Tensor,
    ):
        return self.decode_next_token(idx,top_k,y_len,y,xy_pos)

import time

def export_prov2(
    gpt_path,
    vits_path,
    version,
    ref_audio_path,
    ref_text,
    output_path,
    export_bert_and_ssl=False,
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
        "这是一个简单的示例，真没想到这么简单就完成了。真的神奇。可能这就是狐狸吧.你觉得狐狸神奇吗？", "auto", "v2"
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
    # stream_t2s = torch.jit.script(stream_t2s)

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

    y_len, y, xy_pos = stream_t2s.pre_infer(prompts, ref_seq, text_seq, ref_bert, text_bert, top_k)
    idx = 1
    audio_index = 0
    last_idx = 0
    audios = []
    print("y.shape:", y.shape)
    while True:
        y, xy_pos, stop = stream_t2s(idx, top_k, y_len, y, xy_pos)
        # print("y.shape:", y.shape)

        # 玄学这档子事说不清楚
        if (y[0,-1] < 60 and idx-last_idx > 25) or stop:
            audio = vits.vq_model(y[:,-idx:-1].unsqueeze(0), text_seq, refer, speed=1.0, sv_emb=sv_emb)[0, 0]
            if last_idx == 0:
                audio = audio[:-640]
                et = time.time()
            else:
                if stop:
                    audio = audio[last_idx*1280 -640:]
                else:
                    audio = audio[last_idx*1280 -640:-640]
            print(y[:,-idx+last_idx:])
            last_idx = idx
            # print(f'write {output_path}/out_{audio_index}')
            # soundfile.write(f"{output_path}/out_{audio_index}.wav", audio.float().detach().cpu().numpy(), 32000)
            audio_index+=1
            audios.append(audio)

        idx+=1
        # print(idx,'/',1500 , y.shape, y[0,-1].item(), stop)
        if idx>1500:
            break

        if stop:
            break

    at = time.time()

    for (i,a) in enumerate(audios):
        print(f'write {output_path}/out_{i}')
        soundfile.write(f"{output_path}/out_{i}.wav", a.float().detach().cpu().numpy(), 32000)
        
    print("final,",audio_index)
    print(f"frist token: {et - st:.4f} seconds")
    print(f"all token: {at - st:.4f} seconds")
    audio = vits.vq_model(y[:,-idx:].unsqueeze(0), text_seq, refer, speed=1.0, sv_emb=sv_emb)[0, 0]
    soundfile.write(f"{output_path}/out_final.wav", audio.float().detach().cpu().numpy(), 32000)
    audio = torch.cat(audios, dim=0)
    soundfile.write(f"{output_path}/out.wav", audio.float().detach().cpu().numpy(), 32000)


if __name__ == "__main__":
    with torch.no_grad():
        export_prov2(
            gpt_path="GPT_SoVITS/pretrained_models/s1v3.ckpt",
            vits_path="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
            version="v2Pro",
            ref_audio_path="output/denoise_opt/ht/ht.mp4_0000026560_0000147200.wav",
            ref_text="真的，这件衣服才配得上本小姐嘛",
            output_path="streaming",
            export_bert_and_ssl=True,
            device="cuda",
            is_half=True,
        )
