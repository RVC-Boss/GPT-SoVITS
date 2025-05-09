import os
from export_torch_script import (
    T2SModel,
    get_raw_t2s_model,
    resamplex,
    spectrogram_torch,
)
from f5_tts.model.backbones.dit import DiT
from inference_webui import get_phones_and_bert
import librosa
from module import commons
from module.mel_processing import mel_spectrogram_torch
from module.models_onnx import CFM, SynthesizerTrnV3
import numpy as np
import torch._dynamo.config
import torchaudio
import logging
import uvicorn
import torch
import soundfile
from librosa.filters import mel as librosa_mel_fn


from inference_webui import get_spepc, norm_spec, resample, ssl_model

logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logger = logging.getLogger("uvicorn")

is_half = True
device = "cuda" if torch.cuda.is_available() else "cpu"
now_dir = os.getcwd()


class MelSpectrgram(torch.nn.Module):
    def __init__(
        self,
        dtype,
        device,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        super().__init__()
        self.hann_window = torch.hann_window(1024).to(device=device, dtype=dtype)
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        self.mel_basis = torch.from_numpy(mel).to(dtype=dtype, device=device)
        self.n_fft: int = n_fft
        self.hop_size: int = hop_size
        self.win_size: int = win_size
        self.center: bool = center

    def forward(self, y):
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis, spec)
        # spec = spectral_normalize_torch(spec)
        spec = torch.log(torch.clamp(spec, min=1e-5))
        return spec


class ExportDitBlocks(torch.nn.Module):
    def __init__(self, dit: DiT):
        super().__init__()
        self.transformer_blocks = dit.transformer_blocks
        self.norm_out = dit.norm_out
        self.proj_out = dit.proj_out
        self.depth = dit.depth

    def forward(self, x, t, mask, rope):
        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=(rope, 1.0))
        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output


class ExportDitEmbed(torch.nn.Module):
    def __init__(self, dit: DiT):
        super().__init__()
        self.time_embed = dit.time_embed
        self.d_embed = dit.d_embed
        self.text_embed = dit.text_embed
        self.input_embed = dit.input_embed
        self.rotary_embed = dit.rotary_embed
        self.rotary_embed.inv_freq.to(device)

    def forward(
        self,
        x0: torch.Tensor,  # nosied input audio  # noqa: F722
        cond0: torch.Tensor,  # masked cond audio  # noqa: F722
        x_lens: torch.Tensor,
        time: torch.Tensor,  # time step  # noqa: F821 F722
        dt_base_bootstrap: torch.Tensor,
        text0: torch.Tensor,  # noqa: F722#####condition feature
    ):
        x = x0.transpose(2, 1)
        cond = cond0.transpose(2, 1)
        text = text0.transpose(2, 1)
        mask = commons.sequence_mask(x_lens, max_length=x.size(1)).to(x.device)

        t = self.time_embed(time) + self.d_embed(dt_base_bootstrap)
        text_embed = self.text_embed(text, x.shape[1])
        rope_t = torch.arange(x.shape[1], device=device)
        rope, _ = self.rotary_embed(rope_t)
        x = self.input_embed(x, cond, text_embed)
        return x, t, mask, rope


class ExportDiT(torch.nn.Module):
    def __init__(self, dit: DiT):
        super().__init__()
        if dit != None:
            self.embed = ExportDitEmbed(dit)
            self.blocks = ExportDitBlocks(dit)
        else:
            self.embed = None
            self.blocks = None

    def forward(  # x, prompt_x, x_lens, t, style,cond
        self,  # d is channel,n is T
        x0: torch.Tensor,  # nosied input audio  # noqa: F722
        cond0: torch.Tensor,  # masked cond audio  # noqa: F722
        x_lens: torch.Tensor,
        time: torch.Tensor,  # time step  # noqa: F821 F722
        dt_base_bootstrap: torch.Tensor,
        text0: torch.Tensor,  # noqa: F722#####condition feature
    ):
        x, t, mask, rope = self.embed(x0, cond0, x_lens, time, dt_base_bootstrap, text0)
        output = self.blocks(x, t, mask, rope)
        return output


class ExportCFM(torch.nn.Module):
    def __init__(self, cfm: CFM):
        super().__init__()
        self.cfm = cfm

    def forward(
        self,
        fea_ref: torch.Tensor,
        fea_todo_chunk: torch.Tensor,
        mel2: torch.Tensor,
        sample_steps: torch.LongTensor,
    ):
        T_min = fea_ref.size(2)
        fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
        cfm_res = self.cfm(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps)
        cfm_res = cfm_res[:, :, mel2.shape[2] :]
        mel2 = cfm_res[:, :, -T_min:]
        fea_ref = fea_todo_chunk[:, :, -T_min:]
        return cfm_res, fea_ref, mel2


mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)

spec_min = -12
spec_max = 2


@torch.jit.script
def norm_spec(x):
    spec_min = -12
    spec_max = 2
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x):
    spec_min = -12
    spec_max = 2
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


class ExportGPTSovitsHalf(torch.nn.Module):
    def __init__(self, hps, t2s_m: T2SModel, vq_model: SynthesizerTrnV3):
        super().__init__()
        self.hps = hps
        self.t2s_m = t2s_m
        self.vq_model = vq_model
        self.mel2 = MelSpectrgram(
            dtype=torch.float32,
            device=device,
            n_fft=1024,
            num_mels=100,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=None,
            center=False,
        )
        # self.dtype = dtype
        self.filter_length: int = hps.data.filter_length
        self.sampling_rate: int = hps.data.sampling_rate
        self.hop_length: int = hps.data.hop_length
        self.win_length: int = hps.data.win_length

    def forward(
        self,
        ssl_content,
        ref_audio_32k: torch.FloatTensor,
        phoneme_ids0,
        phoneme_ids1,
        bert1,
        bert2,
        top_k,
    ):
        refer = spectrogram_torch(
            ref_audio_32k,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        ).to(ssl_content.dtype)

        codes = self.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0)
        # print('extract_latent',codes.shape,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        pred_semantic = self.t2s_m(prompt, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k)
        # print('t2s_m',pred_semantic.shape,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        ge = self.vq_model.create_ge(refer)
        # print('create_ge',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        prompt_ = prompt.unsqueeze(0)
        fea_ref = self.vq_model(prompt_, phoneme_ids0, ge)
        # print('fea_ref',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # print(prompt_.shape, phoneme_ids0.shape, ge.shape)
        # print(fea_ref.shape)

        ref_24k = resamplex(ref_audio_32k, 32000, 24000)
        mel2 = norm_spec(self.mel2(ref_24k)).to(ssl_content.dtype)
        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        if T_min > 468:
            mel2 = mel2[:, :, -468:]
            fea_ref = fea_ref[:, :, -468:]
            T_min = 468

        fea_todo = self.vq_model(pred_semantic, phoneme_ids1, ge)
        # print('fea_todo',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # print(pred_semantic.shape, phoneme_ids1.shape, ge.shape)
        # print(fea_todo.shape)

        return fea_ref, fea_todo, mel2


class GPTSoVITSV3(torch.nn.Module):
    def __init__(self, gpt_sovits_half, cfm, bigvgan):
        super().__init__()
        self.gpt_sovits_half = gpt_sovits_half
        self.cfm = cfm
        self.bigvgan = bigvgan

    def forward(
        self,
        ssl_content,
        ref_audio_32k: torch.FloatTensor,
        phoneme_ids0: torch.LongTensor,
        phoneme_ids1: torch.LongTensor,
        bert1,
        bert2,
        top_k: torch.LongTensor,
        sample_steps: torch.LongTensor,
    ):
        # current_time = datetime.now()
        # print("gpt_sovits_half",current_time.strftime("%Y-%m-%d %H:%M:%S"))
        fea_ref, fea_todo, mel2 = self.gpt_sovits_half(
            ssl_content, ref_audio_32k, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k
        )
        chunk_len = 934 - fea_ref.shape[2]
        wav_gen_list = []
        idx = 0
        wav_gen_length = fea_todo.shape[2] * 256
        while 1:
            # current_time = datetime.now()
            # print("idx:",idx,current_time.strftime("%Y-%m-%d %H:%M:%S"))
            fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break

            # 因为导出的模型在不同shape时会重新编译还是怎么的，会卡顿10s这样，
            # 所以在这里补0让他shape维持不变
            # 但是这样会导致生成的音频长度不对，所以在最后截取一下。
            # 经过 bigvgan 之后音频长度就是 fea_todo.shape[2] * 256
            complete_len = chunk_len - fea_todo_chunk.shape[-1]
            if complete_len != 0:
                fea_todo_chunk = torch.cat(
                    [
                        fea_todo_chunk,
                        torch.zeros(1, 512, complete_len).to(fea_todo_chunk.device).to(fea_todo_chunk.dtype),
                    ],
                    2,
                )

            cfm_res, fea_ref, mel2 = self.cfm(fea_ref, fea_todo_chunk, mel2, sample_steps)
            idx += chunk_len

            cfm_res = denorm_spec(cfm_res)
            bigvgan_res = self.bigvgan(cfm_res)
            wav_gen_list.append(bigvgan_res)

        wav_gen = torch.cat(wav_gen_list, 2)
        return wav_gen[0][0][:wav_gen_length]


def init_bigvgan():
    global bigvgan_model
    from BigVGAN import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False,
    )  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)


class Sovits:
    def __init__(self, vq_model: SynthesizerTrnV3, cfm: CFM, hps):
        self.vq_model = vq_model
        self.hps = hps
        cfm.estimator = ExportDiT(cfm.estimator)
        self.cfm = cfm


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


from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new


def get_sovits_weights(sovits_path):
    path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
    is_exist_s2gv3 = os.path.exists(path_sovits_v3)

    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    if if_lora_v3 == True and is_exist_s2gv3 == False:
        logger.info("SoVITS V3 底模缺失，无法加载相应 LoRA 权重")

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"  # v3model,v2sybomls
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"

    if model_version == "v3":
        hps.model.version = "v3"

    logger.info(f"hps: {hps}")

    vq_model = SynthesizerTrnV3(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    # init_bigvgan()
    model_version = hps.model.version
    logger.info(f"模型版本: {model_version}")

    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    vq_model.eval()

    cfm = vq_model.cfm
    del vq_model.cfm

    sovits = Sovits(vq_model, cfm, hps)
    return sovits


logger.info(f"torch version {torch.__version__}")
# ssl_model = cnhubert.get_model()
# if is_half:
#     ssl_model = ssl_model.half().to(device)
# else:
#     ssl_model = ssl_model.to(device)


def export_cfm(
    e_cfm: ExportCFM,
    mu: torch.Tensor,
    x_lens: torch.LongTensor,
    prompt: torch.Tensor,
    n_timesteps: torch.IntTensor,
    temperature=1.0,
):
    cfm = e_cfm.cfm

    B, T = mu.size(0), mu.size(1)
    x = torch.randn([B, cfm.in_channels, T], device=mu.device, dtype=mu.dtype) * temperature
    print("x:", x.shape, x.dtype)
    prompt_len = prompt.size(-1)
    prompt_x = torch.zeros_like(x, dtype=mu.dtype)
    prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
    x[..., :prompt_len] = 0.0
    mu = mu.transpose(2, 1)

    ntimestep = int(n_timesteps)

    t = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    d = torch.tensor(1.0 / ntimestep, dtype=x.dtype, device=x.device)

    t_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * t
    d_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * d

    print(
        "cfm input shapes:",
        x.shape,
        prompt_x.shape,
        x_lens.shape,
        t_tensor.shape,
        d_tensor.shape,
        mu.shape,
    )

    print("cfm input dtypes:", x.dtype, prompt_x.dtype, x_lens.dtype, t_tensor.dtype, d_tensor.dtype, mu.dtype)

    estimator: ExportDiT = torch.jit.trace(
        cfm.estimator,
        optimize=True,
        example_inputs=(x, prompt_x, x_lens, t_tensor, d_tensor, mu),
    )
    estimator.save("onnx/ad/estimator.pt")
    # torch.onnx.export(
    #     cfm.estimator,
    #     (x, prompt_x, x_lens, t_tensor, d_tensor, mu),
    #     "onnx/ad/dit.onnx",
    #     input_names=["x", "prompt_x", "x_lens", "t", "d", "mu"],
    #     output_names=["output"],
    #     dynamic_axes={
    #         "x": [2],
    #         "prompt_x": [2],
    #         "mu": [2],
    #     },
    # )
    print("save estimator ok")
    cfm.estimator = estimator
    export_cfm = torch.jit.script(e_cfm)
    export_cfm.save("onnx/ad/cfm.pt")
    # sovits.cfm = cfm
    # cfm.save("onnx/ad/cfm.pt")
    return export_cfm


def export():
    sovits = get_sovits_weights("GPT_SoVITS/pretrained_models/s2Gv3.pth")

    init_bigvgan()

    dict_s1 = torch.load("GPT_SoVITS/pretrained_models/s1v3.ckpt")
    raw_t2s = get_raw_t2s_model(dict_s1).to(device)
    print("#### get_raw_t2s_model ####")
    print(raw_t2s.config)

    if is_half:
        raw_t2s = raw_t2s.half().to(device)

    t2s_m = T2SModel(raw_t2s)
    t2s_m.eval()
    script_t2s = torch.jit.script(t2s_m).to(device)

    hps = sovits.hps
    ref_wav_path = "onnx/ad/ref.wav"
    speed = 1.0
    sample_steps = 32
    dtype = torch.float16 if is_half == True else torch.float32
    refer = get_spepc(hps, ref_wav_path).to(device).to(dtype)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )

    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)

        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        codes = sovits.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)

    phones1, bert1, norm_text1 = get_phones_and_bert(
        "你这老坏蛋，我找了你这么久，真没想到在这里找到你。他说。", "all_zh", "v3"
    )
    phones2, bert2, norm_text2 = get_phones_and_bert(
        "这是一个简单的示例，真没想到这么简单就完成了。The King and His Stories.Once there was a king. He likes to write stories, but his stories were not good. As people were afraid of him, they all said his stories were good.After reading them, the writer at once turned to the soldiers and said: Take me back to prison, please.",
        "auto",
        "v3",
    )
    phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
    phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)

    # codes = sovits.vq_model.extract_latent(ssl_content)
    # prompt_semantic = codes[0, 0]
    # prompts = prompt_semantic.unsqueeze(0)

    top_k = torch.LongTensor([15]).to(device)
    print("topk", top_k)

    bert1 = bert1.T.to(device)
    bert2 = bert2.T.to(device)
    print(
        prompt.dtype,
        phoneme_ids0.dtype,
        phoneme_ids1.dtype,
        bert1.dtype,
        bert2.dtype,
        top_k.dtype,
    )
    print(
        prompt.shape,
        phoneme_ids0.shape,
        phoneme_ids1.shape,
        bert1.shape,
        bert2.shape,
        top_k.shape,
    )
    pred_semantic = t2s_m(prompt, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k)

    ge = sovits.vq_model.create_ge(refer)
    prompt_ = prompt.unsqueeze(0)

    torch._dynamo.mark_dynamic(prompt_, 2)
    torch._dynamo.mark_dynamic(phoneme_ids0, 1)

    fea_ref = sovits.vq_model(prompt_, phoneme_ids0, ge)

    inputs = {
        "forward": (prompt_, phoneme_ids0, ge),
        "extract_latent": ssl_content,
        "create_ge": refer,
    }

    trace_vq_model = torch.jit.trace_module(sovits.vq_model, inputs, optimize=True)
    trace_vq_model.save("onnx/ad/vq_model.pt")

    print(fea_ref.shape, fea_ref.dtype, ge.shape)
    print(prompt_.shape, phoneme_ids0.shape, ge.shape)

    # vq_model = torch.jit.trace(
    #     sovits.vq_model,
    #     optimize=True,
    #     # strict=False,
    #     example_inputs=(prompt_, phoneme_ids0, ge),
    # )
    # vq_model = sovits.vq_model
    vq_model = trace_vq_model

    gpt_sovits_half = ExportGPTSovitsHalf(sovits.hps, script_t2s, trace_vq_model)
    torch.jit.script(gpt_sovits_half).save("onnx/ad/gpt_sovits_v3_half.pt")

    ref_audio, sr = torchaudio.load(ref_wav_path)
    ref_audio = ref_audio.to(device).float()
    if ref_audio.shape[0] == 2:
        ref_audio = ref_audio.mean(0).unsqueeze(0)
    if sr != 24000:
        ref_audio = resample(ref_audio, sr)
    # mel2 = mel_fn(ref_audio)
    mel2 = norm_spec(mel_fn(ref_audio))
    T_min = min(mel2.shape[2], fea_ref.shape[2])
    fea_ref = fea_ref[:, :, :T_min]
    print("fea_ref:", fea_ref.shape, T_min)
    if T_min > 468:
        mel2 = mel2[:, :, -468:]
        fea_ref = fea_ref[:, :, -468:]
        T_min = 468
    chunk_len = 934 - T_min
    mel2 = mel2.to(dtype)

    # fea_todo, ge = sovits.vq_model(pred_semantic,y_lengths, phoneme_ids1, ge)
    fea_todo = vq_model(pred_semantic, phoneme_ids1, ge)

    cfm_resss = []
    idx = 0
    sample_steps = torch.LongTensor([sample_steps]).to(device)
    export_cfm_ = ExportCFM(sovits.cfm)
    while 1:
        print("idx:", idx)
        fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
        if fea_todo_chunk.shape[-1] == 0:
            break

        print(
            "export_cfm:",
            fea_ref.shape,
            fea_todo_chunk.shape,
            mel2.shape,
            sample_steps.shape,
        )
        if idx == 0:
            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
            export_cfm_ = export_cfm(
                export_cfm_,
                fea,
                torch.LongTensor([fea.size(1)]).to(fea.device),
                mel2,
                sample_steps,
            )
            # torch.onnx.export(
            #     export_cfm_,
            #     (
            #         fea_ref,
            #         fea_todo_chunk,
            #         mel2,
            #         sample_steps,
            #     ),
            #     "onnx/ad/cfm.onnx",
            #     input_names=["fea_ref", "fea_todo_chunk", "mel2", "sample_steps"],
            #     output_names=["cfm_res", "fea_ref_", "mel2_"],
            #     dynamic_axes={
            #         "fea_ref": [2],
            #         "fea_todo_chunk": [2],
            #         "mel2": [2],
            #     },
            # )

        idx += chunk_len

        cfm_res, fea_ref, mel2 = export_cfm_(fea_ref, fea_todo_chunk, mel2, sample_steps)
        cfm_resss.append(cfm_res)
        continue

    cmf_res = torch.cat(cfm_resss, 2)
    cmf_res = denorm_spec(cmf_res).to(device)
    print("cmf_res:", cmf_res.shape, cmf_res.dtype)
    with torch.inference_mode():
        cmf_res_rand = torch.randn(1, 100, 934).to(device).to(dtype)
        torch._dynamo.mark_dynamic(cmf_res_rand, 2)
        bigvgan_model_ = torch.jit.trace(bigvgan_model, optimize=True, example_inputs=(cmf_res_rand,))
        bigvgan_model_.save("onnx/ad/bigvgan_model.pt")
        wav_gen = bigvgan_model(cmf_res)
        print("wav_gen:", wav_gen.shape, wav_gen.dtype)
        audio = wav_gen[0][0].cpu().detach().numpy()

    sr = 24000
    soundfile.write("out.export.wav", (audio * 32768).astype(np.int16), sr)


from datetime import datetime


def test_export(
    todo_text,
    gpt_sovits_v3_half,
    cfm,
    bigvgan,
    output,
):
    # hps = sovits.hps
    ref_wav_path = "onnx/ad/ref.wav"
    speed = 1.0
    sample_steps = 8

    dtype = torch.float16 if is_half == True else torch.float32

    zero_wav = np.zeros(
        int(16000 * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )

    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)

        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()

    ref_audio_32k, _ = librosa.load(ref_wav_path, sr=32000)
    ref_audio_32k = torch.from_numpy(ref_audio_32k).unsqueeze(0).to(device).float()

    phones1, bert1, norm_text1 = get_phones_and_bert(
        "你这老坏蛋，我找了你这么久，真没想到在这里找到你。他说。", "all_zh", "v3"
    )
    phones2, bert2, norm_text2 = get_phones_and_bert(
        todo_text,
        "zh",
        "v3",
    )
    phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
    phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)

    bert1 = bert1.T.to(device)
    bert2 = bert2.T.to(device)
    top_k = torch.LongTensor([15]).to(device)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("start inference %s", current_time)
    print(
        ssl_content.shape,
        ref_audio_32k.shape,
        phoneme_ids0.shape,
        phoneme_ids1.shape,
        bert1.shape,
        bert2.shape,
        top_k.shape,
    )
    fea_ref, fea_todo, mel2 = gpt_sovits_v3_half(
        ssl_content, ref_audio_32k, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k
    )
    chunk_len = 934 - fea_ref.shape[2]
    print(fea_ref.shape, fea_todo.shape, mel2.shape)

    cfm_resss = []
    sample_steps = torch.LongTensor([sample_steps])
    idx = 0
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("start cfm %s", current_time)
    wav_gen_length = fea_todo.shape[2] * 256

    while 1:
        current_time = datetime.now()
        print("idx:", idx, current_time.strftime("%Y-%m-%d %H:%M:%S"))
        fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
        if fea_todo_chunk.shape[-1] == 0:
            break

        complete_len = chunk_len - fea_todo_chunk.shape[-1]
        if complete_len != 0:
            fea_todo_chunk = torch.cat([fea_todo_chunk, torch.zeros(1, 512, complete_len).to(device).to(dtype)], 2)

        cfm_res, fea_ref, mel2 = cfm(fea_ref, fea_todo_chunk, mel2, sample_steps)
        # if complete_len > 0 :
        #     cfm_res = cfm_res[:, :, :-complete_len]
        #     fea_ref = fea_ref[:, :, :-complete_len]
        #     mel2 = mel2[:, :, :-complete_len]

        idx += chunk_len

        current_time = datetime.now()
        print("cfm end", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        cfm_res = denorm_spec(cfm_res).to(device)
        bigvgan_res = bigvgan(cfm_res)
        cfm_resss.append(bigvgan_res)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("start bigvgan %s", current_time)
    wav_gen = torch.cat(cfm_resss, 2)
    # cmf_res = denorm_spec(cmf_res)
    # cmf_res = cmf_res.to(device)
    # print("cmf_res:", cmf_res.shape)

    # cmf_res = torch.cat([cmf_res,torch.zeros([1,100,2000-cmf_res.size(2)],device=device,dtype=cmf_res.dtype)], 2)

    # wav_gen = bigvgan(cmf_res)
    print("wav_gen:", wav_gen.shape, wav_gen.dtype)
    wav_gen = wav_gen[:, :, :wav_gen_length]

    audio = wav_gen[0][0].cpu().detach().numpy()
    logger.info("end bigvgan %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sr = 24000
    soundfile.write(output, (audio * 32768).astype(np.int16), sr)


def test_export1(
    todo_text,
    gpt_sovits_v3,
    output,
):
    # hps = sovits.hps
    ref_wav_path = "onnx/ad/ref.wav"
    speed = 1.0
    sample_steps = torch.LongTensor([16])

    dtype = torch.float16 if is_half == True else torch.float32

    zero_wav = np.zeros(
        int(24000 * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )

    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)

        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        print("ssl_content:", ssl_content.shape, ssl_content.dtype)

    ref_audio_32k, _ = librosa.load(ref_wav_path, sr=32000)
    ref_audio_32k = torch.from_numpy(ref_audio_32k).unsqueeze(0).to(device).float()

    phones1, bert1, norm_text1 = get_phones_and_bert(
        "你这老坏蛋，我找了你这么久，真没想到在这里找到你。他说。", "all_zh", "v3"
    )
    phones2, bert2, norm_text2 = get_phones_and_bert(
        todo_text,
        "zh",
        "v3",
    )
    phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
    phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)

    bert1 = bert1.T.to(device)
    bert2 = bert2.T.to(device)
    top_k = torch.LongTensor([15]).to(device)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("start inference %s", current_time)
    print(
        ssl_content.shape,
        ref_audio_32k.shape,
        phoneme_ids0.shape,
        phoneme_ids1.shape,
        bert1.shape,
        bert2.shape,
        top_k.shape,
    )
    wav_gen = gpt_sovits_v3(ssl_content, ref_audio_32k, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k, sample_steps)
    print("wav_gen:", wav_gen.shape, wav_gen.dtype)

    wav_gen = torch.cat([wav_gen, zero_wav_torch], 0)

    audio = wav_gen.cpu().detach().numpy()
    logger.info("end bigvgan %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sr = 24000
    soundfile.write(output, (audio * 32768).astype(np.int16), sr)


import time


def test_():
    sovits = get_sovits_weights("GPT_SoVITS/pretrained_models/s2Gv3.pth")

    # cfm = ExportCFM(sovits.cfm)
    # cfm.cfm.estimator = dit
    sovits.cfm = None

    cfm = torch.jit.load("onnx/ad/cfm.pt", map_location=device)
    # cfm = torch.jit.optimize_for_inference(cfm)
    cfm = cfm.half().to(device)

    cfm.eval()

    logger.info("cfm ok")

    dict_s1 = torch.load("GPT_SoVITS/pretrained_models/s1v3.ckpt")
    # v2 的 gpt 也可以用
    # dict_s1 = torch.load("GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
    raw_t2s = get_raw_t2s_model(dict_s1).to(device)
    print("#### get_raw_t2s_model ####")
    print(raw_t2s.config)
    if is_half:
        raw_t2s = raw_t2s.half().to(device)
    t2s_m = T2SModel(raw_t2s).half().to(device)
    t2s_m.eval()
    t2s_m = torch.jit.script(t2s_m)
    t2s_m.eval()
    # t2s_m.top_k = 15
    logger.info("t2s_m ok")

    vq_model: torch.jit.ScriptModule = torch.jit.load("onnx/ad/vq_model.pt", map_location=device)
    # vq_model = torch.jit.optimize_for_inference(vq_model)
    # vq_model = vq_model.half().to(device)
    vq_model.eval()
    # vq_model = sovits.vq_model
    logger.info("vq_model ok")

    # gpt_sovits_v3_half = torch.jit.load("onnx/ad/gpt_sovits_v3_half.pt")
    # gpt_sovits_v3_half = torch.jit.optimize_for_inference(gpt_sovits_v3_half)
    # gpt_sovits_v3_half = gpt_sovits_v3_half.half()
    # gpt_sovits_v3_half = gpt_sovits_v3_half.cuda()
    # gpt_sovits_v3_half.eval()
    gpt_sovits_v3_half = ExportGPTSovitsHalf(sovits.hps, t2s_m, vq_model)
    logger.info("gpt_sovits_v3_half ok")

    # init_bigvgan()
    # global bigvgan_model
    bigvgan_model = torch.jit.load("onnx/ad/bigvgan_model.pt")
    # bigvgan_model = torch.jit.optimize_for_inference(bigvgan_model)
    bigvgan_model = bigvgan_model.half()
    bigvgan_model = bigvgan_model.cuda()
    bigvgan_model.eval()

    logger.info("bigvgan ok")

    gpt_sovits_v3 = GPTSoVITSV3(gpt_sovits_v3_half, cfm, bigvgan_model)
    gpt_sovits_v3 = torch.jit.script(gpt_sovits_v3)
    gpt_sovits_v3.save("onnx/ad/gpt_sovits_v3.pt")
    gpt_sovits_v3 = gpt_sovits_v3.half().to(device)
    gpt_sovits_v3.eval()
    print("save gpt_sovits_v3 ok")

    time.sleep(5)
    # print("thread:", torch.get_num_threads())
    # print("thread:", torch.get_num_interop_threads())
    # torch.set_num_interop_threads(1)
    # torch.set_num_threads(1)

    test_export1(
        "汗流浃背了呀!老弟~ My uncle has two dogs. One is big and the other is small. He likes them very much. He often plays with them. He takes them for a walk every day. He says they are his good friends. He is very happy with them. 最后还是我得了 MVP....",
        gpt_sovits_v3,
        "out.wav",
    )

    test_export1(
        "你小子是什么来路.汗流浃背了呀!老弟~ My uncle has two dogs. He is very happy with them. 最后还是我得了 MVP!",
        gpt_sovits_v3,
        "out2.wav",
    )

    # test_export(
    #     "汗流浃背了呀!老弟~ My uncle has two dogs. One is big and the other is small. He likes them very much. He often plays with them. He takes them for a walk every day. He says they are his good friends. He is very happy with them. 最后还是我得了 MVP. 哈哈哈...",
    #     gpt_sovits_v3_half,
    #     cfm,
    #     bigvgan_model,
    #     "out2.wav",
    # )


def test_export_gpt_sovits_v3():
    gpt_sovits_v3 = torch.jit.load("onnx/ad/gpt_sovits_v3.pt", map_location=device)
    # test_export1(
    #     "汗流浃背了呀!老弟~ My uncle has two dogs. One is big and the other is small. He likes them very much. He often plays with them. He takes them for a walk every day. He says they are his good friends. He is very happy with them. 最后还是我得了 MVP....",
    #     gpt_sovits_v3,
    #     "out3.wav",
    # )
    # test_export1(
    #     "你小子是什么来路.汗流浃背了呀!老弟~ My uncle has two dogs. He is very happy with them. 最后还是我得了 MVP!",
    #     gpt_sovits_v3,
    #     "out4.wav",
    # )
    test_export1(
        "风萧萧兮易水寒，壮士一去兮不复还.",
        gpt_sovits_v3,
        "out5.wav",
    )


with torch.no_grad():
    # export()
    test_()
    # test_export_gpt_sovits_v3()
