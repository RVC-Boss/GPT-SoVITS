import argparse
import contextlib
import gc
import logging
import os
import re
import traceback
import warnings
from functools import partial
from pathlib import Path
from time import perf_counter as ttime
from typing import Any

import gradio as gr
import librosa
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForMaskedLM, AutoTokenizer

from config import (
    change_choices,
    get_dtype,
    get_weights_names,
    pretrained_sovits_name,
)
from config import (
    infer_device as default_device,
)
from GPT_SoVITS.Accelerate import MLX, PyTorch, T2SEngineProtocol, T2SRequest, backends
from GPT_SoVITS.Accelerate.logger import console, timer
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
from GPT_SoVITS.process_ckpt import inspect_version
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from tools.i18n.i18n import I18nAuto, scan_language_list
from tools.my_utils import DictToAttrRecursive

warnings.filterwarnings(
    "ignore", message="MPS: The constant padding of more than 3 dimensions is not currently supported natively."
)
warnings.filterwarnings("ignore", message=".*ComplexHalf support is experimental.*")

logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


_LANG_RE = re.compile(r"^[a-z]{2}[_-][A-Z]{2}$")


def lang_type(text: str) -> str:
    if text == "Auto":
        return text
    if not _LANG_RE.match(text):
        raise argparse.ArgumentTypeError(f"Unspported Format: {text}, Expected ll_CC/ll-CC")
    ll, cc = re.split(r"[_-]", text)
    language = f"{ll}_{cc}"
    if language in scan_language_list():
        return language
    else:
        return "Auto"


def none_or_str(value: str):
    if value == "None":
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inference_webui",
        description=f"python -s -m GPT_SoVITS.inference_webui zh_CN -b {backends[-1]}",
    )
    p.add_argument(
        "language",
        nargs="?",
        default="Auto",
        type=lang_type,
        help="Language Code, Such as zh_CN, en-US",
    )
    p.add_argument(
        "--backends",
        "-b",
        choices=backends,
        default=backends[-1],
        help="AR Inference Backend",
        required=False,
    )
    p.add_argument(
        "--quantization",
        "-q",
        default="None",
        choices=MLX.quantization_methods_mlx + PyTorch.quantization_methods_torch,
        type=none_or_str,
        help="Quantization Method",
        required=False,
    )
    p.add_argument(
        "--device",
        "-d",
        default=str(default_device),
        help="Inference Device",
        required=False,
    )
    p.add_argument(
        "--port",
        "-p",
        default=9872,
        type=int,
        help="WebUI Binding Port",
        required=False,
    )
    p.add_argument(
        "--share",
        "-s",
        default=False,
        action="store_true",
        help="Gradio Share Link",
        required=False,
    )
    p.add_argument(
        "--cnhubert",
        default="GPT_SoVITS/pretrained_models/chinese-hubert-base",
        help="CNHuBERT Pretrain",
        required=False,
    )
    p.add_argument(
        "--bert",
        default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        help="BERT Pretrain",
        required=False,
    )
    p.add_argument(
        "--gpt",
        default="",
        help="GPT Model",
        required=False,
    )
    p.add_argument(
        "--sovits",
        default="",
        help="SoVITS Model",
        required=False,
    )

    return p


args = build_parser().parse_args()

hps: Any = None
vq_model: SynthesizerTrn | SynthesizerTrnV3 | None = None
t2s_engine: T2SEngineProtocol | None = None

version = model_version = "v2"
path_sovits_v3 = pretrained_sovits_name["v3"]
path_sovits_v4 = pretrained_sovits_name["v4"]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)

cnhubert_base_path = str(args.cnhubert)
bert_path = str(args.bert)
infer_ttswebui = int(args.port)
is_share = bool(args.share)


i18n = I18nAuto(language=args.language)
ar_backend: str = args.backends
change_choices_i18n = partial(change_choices, i18n=i18n)

SoVITS_names, GPT_names = get_weights_names(i18n)


dict_language_v1 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别
    i18n("日英混合"): "ja",  # 按日英混合识别
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("粤语"): "all_yue",  # 全部按粤语识别
    i18n("韩文"): "all_ko",  # 全部按韩文识别
    i18n("中英混合"): "zh",
    i18n("日英混合"): "ja",
    i18n("粤英混合"): "yue",
    i18n("韩英混合"): "ko",
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种
}
dict_language = dict_language_v1 if version == "v1" else dict_language_v2

punctuation = set(["!", "?", "…", ",", ".", "-", " "])
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
v3v4set = {"v3", "v4"}

infer_device = torch.device(args.device)
device = infer_device if infer_device.type == "cuda" else torch.device("cpu")

dtype = get_dtype(device.index)
is_half = dtype == torch.float16

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path).to(infer_device, dtype)

cnhubert.cnhubert_base_path = cnhubert_base_path
ssl_model = cnhubert.get_model().to(infer_device, dtype)


def mel_fn(x):
    return mel_spectrogram_torch(
        y=x,
        n_fft=1024,
        num_mels=100,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=None,
        center=False,
    )


gpt_path = str(args.gpt) or GPT_names[0][-1]
sovits_path = str(args.sovits) or SoVITS_names[0][-1]


def get_bert_feature(text, word2ph):
    inputs = tokenizer(text, return_tensors="pt")
    for i in inputs:
        inputs[i] = inputs[i].to(infer_device)
    res = bert_model(**inputs, output_hidden_states=True)
    res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature_t = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature_t.T


def change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
    global vq_model, hps, version, model_version, dict_language
    model_version, version, is_lora, hps, dict_s2 = inspect_version(sovits_path)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    if is_lora is True and is_exist is False:
        info = f"{path_sovits} SoVITS {model_version} {i18n('底模缺失, 无法加载相应 LoRA 权重')}"
        gr.Warning(info)
        raise FileNotFoundError(info)
    dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    visible_sample_steps = visible_inp_refs = None
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = gr.skip(), gr.update(choices=list(dict_language.keys()))
        else:
            prompt_text_update = gr.update(value="")
            prompt_language_update = gr.update(value=i18n("中文"), choices=list(dict_language.keys()))
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = gr.skip(), gr.skip()
        else:
            text_update = gr.update(value="")
            text_language_update = gr.update(value=i18n("中文"), choices=list(dict_language.keys()))

        if model_version in v3v4set:
            visible_sample_steps = True
            visible_inp_refs = False
        else:
            visible_sample_steps = False
            visible_inp_refs = True
        yield (
            prompt_text_update,
            prompt_language_update,
            text_update,
            text_language_update,
            gr.update(
                visible=visible_sample_steps,
                value=32 if model_version == "v3" else 8,
                choices=[4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
            ),
            gr.update(visible=visible_inp_refs),
            gr.update(value=False, interactive=True if model_version not in v3v4set else False),
            gr.update(visible=True if model_version == "v3" else False),
            gr.update(value=i18n("模型加载中, 请等待"), interactive=False),
        )

    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    hps.model.version = model_version
    if model_version not in v3v4set:
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).eval()

    if "pretrained" not in sovits_path:
        if hasattr(vq_model, "enc_q"):
            del vq_model.enc_q

    vq_model.load_state_dict(dict_s2["weight"])

    vq_model = vq_model.to(infer_device, dtype)

    yield (
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.update(value=i18n("合成语音"), interactive=True),
    )


with contextlib.suppress(UnboundLocalError):
    next(change_sovits_weights(sovits_path))


def change_gpt_weights(gpt_path):
    global t2s_engine, config
    if "mlx" in ar_backend.lower():
        t2s_engine = MLX.T2SEngineMLX(
            MLX.T2SEngineMLX.load_decoder(Path(gpt_path), backend=ar_backend, quantize_mode=args.quantization),
            "mx.gpu" if infer_device.type != "cpu" else "mx.cpu",
            dtype=dtype,
        )
        # t2s_engine.decoder_model.compile()
    else:
        t2s_engine = PyTorch.T2SEngineTorch(
            PyTorch.T2SEngineTorch.load_decoder(Path(gpt_path), backend=ar_backend, quantize_mode=args.quantization),
            device,
            dtype=dtype,
        )
        # t2s_engine.decoder_model.compile()


change_gpt_weights(gpt_path)

resample_transform_dict = {}


def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = f"{sr0}-{sr1}-{device}"
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load_with_torchcodec(filename)
    audio = audio.to(device)

    if sr0 != sr1:
        audio = resample(audio, sr0, sr1, device)
    if audio.shape[0] > 1:
        audio = audio.mean(0).unsqueeze(0)

    maxx = float(audio.abs().max())
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro is True:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio


def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half is True else torch.float32,
        ).to(device)

    return bert


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_phones_and_bert(text, language, version, final=False):
    text = re.sub(r" {2,}", " ", text)
    textlist = []
    langlist = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text, "ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text, "ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(dtype), norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


sr_model = None


def audio_sr(audio, sr):
    global sr_model
    if sr_model is None:
        from tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(infer_device, DictToAttrRecursive)
        except FileNotFoundError:
            gr.Warning(i18n("你没有下载超分模型的参数, 因此不进行超分, 如想超分请先参照教程把文件下载好"))
            return audio.cpu().numpy(), sr
    return sr_model(audio, sr)


cache: dict[int, Any] = {}


def get_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut=i18n("不切"),
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free=False,
    speed=1,
    if_freeze=False,
    inp_refs=None,
    sample_steps=8,
    if_sr=False,
    pause_second=0.3,
):
    torch.set_grad_enabled(False)
    debug = os.getenv("DEBUG") == "1"
    ttft_time = ttime()

    if ref_wav_path:
        pass
    else:
        gr.Warning(i18n("请上传参考音频"))
    if text:
        pass
    else:
        gr.Warning(i18n("请填入推理文本"))
    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    if model_version in v3v4set:
        ref_free = False  # s2v3暂不支持ref_free
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
    text = text.strip("\n")

    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half is True else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half is True:
        zero_wav_torch = zero_wav_torch.half().to(infer_device)
    else:
        zero_wav_torch = zero_wav_torch.to(infer_device)
    if not ref_free:
        assert vq_model
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            gr.Warning(i18n("参考音频在3~10秒范围外, 请更换!"))
            raise OSError(i18n("参考音频在3~10秒范围外, 请更换!"))
        wav16k_t = torch.from_numpy(wav16k)
        if is_half is True:
            wav16k_t = wav16k_t.half().to(infer_device)
        else:
            wav16k_t = wav16k_t.to(infer_device)
        wav16k_t = torch.cat([wav16k_t, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k_t.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)
    else:
        prompt = torch.zeros((1, 0)).to(device, torch.int32)

    t1 = ttime()
    t.append(t1 - t0)

    if how_to_cut == i18n("凑四句一切"):
        text = cut1(text)
    elif how_to_cut == i18n("凑50字一切"):
        text = cut2(text)
    elif how_to_cut == i18n("按中文句号。切"):
        text = cut3(text)
    elif how_to_cut == i18n("按英文句号.切"):
        text = cut4(text)
    elif how_to_cut == i18n("按标点符号切"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    # s2v3暂不支持ref_free
    if not ref_free:
        phones1, bert1, _ = get_phones_and_bert(prompt_text, prompt_language, version)
    else:
        phones1, bert1 = [], torch.zeros(1024, 0).to(device, dtype)

    infer_len: list[int] = []
    infer_time: list[float] = []
    assert vq_model

    for i_text, text in enumerate(texts):
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)

        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        if i_text in cache and if_freeze is True:
            pred_semantic = cache[i_text]
        else:
            t2s_request = T2SRequest(
                [all_phoneme_ids.squeeze(0)],
                all_phoneme_len,
                prompt,
                [bert.squeeze(0)],
                valid_length=1,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=1500,
                use_cuda_graph=torch.cuda.is_available(),  # Try to use CUDA Graph for all backend, fallback to normal if not applicapble
                debug=debug,
            )
            assert t2s_engine
            t2s_result = t2s_engine.generate(t2s_request)
            if t2s_result.exception is not None:
                console.print(t2s_result.traceback)
                raise RuntimeError()
            pred_semantic_list = t2s_result.result
            assert pred_semantic_list, t2s_result.traceback
            pred_semantic = pred_semantic_list[0].unsqueeze(0).to(infer_device)
            infer_len.append(t2s_result.total_tokens)
            infer_time.append(t2s_result.infer_speed[-1])

            cache[i_text] = pred_semantic
        t3 = ttime()
        is_v2pro = model_version in {"v2Pro", "v2ProPlus"}

        refers = []
        if inp_refs:
            for path in inp_refs:
                try:  # 这里加上提取sv的逻辑，要么一堆sv一堆refer，要么单个sv单个refer
                    refer, audio_tensor = get_spepc(hps, path.name, dtype, infer_device, is_v2pro)
                    refers.append(refer)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
        if len(refers) == 0:
            refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, infer_device, is_v2pro)
            refers = [refers]
        audio = vq_model.decode(
            pred_semantic,
            torch.LongTensor(phones2).to(infer_device).unsqueeze(0),
            refers,
            speed=speed,
        )[0][0]  # type: ignore

        if i_text == 0:
            ttft_time = ttime() - ttft_time
        max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio = audio / max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)  # zero_wav
        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()

    audio_opt_t = torch.cat(audio_opt, 0)  # np.concatenate
    if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000  # v4
    audio_opt_n = audio_opt_t.cpu().numpy()

    t0 = t[0]
    t1 = sum(t[1::3])
    t2 = sum(t[2::3])
    t3 = sum(t[3::3])

    infer_speed_avg = sum(infer_len) / sum(infer_time) if infer_time else 0
    rtf_value = sum(t) / (audio_opt_n.__len__() / opt_sr)

    console.print(f">> Time Stamps: {t0:.3f}\t{t1:.3f}\t{t2:.3f}\t{t3:.3f}")
    console.print(f">> Infer Speed: {infer_speed_avg:.2f} Token/s")
    console.print(f">> RTF: {rtf_value:.2f}")

    if ttft_time > 2:
        console.print(f">> TTFT: {ttft_time:.3f} s")
    else:
        console.print(f">> TTFT: {ttft_time * 1000:.3f} ms")

    yield opt_sr, (audio_opt_n * 32767).astype(np.int16)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx: list[int | None] = list(range(0, len(inps) + 1, 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50:  # 如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = inp.strip("。").split("。")
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


a = get_tts_wav(
    "/Users/XXXXRT/Desktop/参考/不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊.wav",
    "不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊",
    i18n("中文"),
    "我在我青春韶华的时候遇到了你，还记得刚刚开学的时候，那是第一次见你，我和我朋友在楼道间打闹的时候无意间瞟到了你正在学习时的侧颜",
    i18n("中文"),
)

next(a)

timer.clear()

a = get_tts_wav(
    "/Users/XXXXRT/Desktop/参考/Cream去能理解很多人的想法时,既然已经被这样想了,没有挽回的余地了.wav",
    "去能理解很多人的想法时,既然已经被这样想了,没有挽回的余地了",
    i18n("中文"),
    "我在我青春韶华的时候遇到了你，还记得刚刚开学的时候，那是第一次见你，我和我朋友在楼道间打闹的时候无意间瞟到了你正在学习时的侧颜",
    i18n("中文"),
)


next(a)

timer.summary()

a = get_tts_wav(
    "/Users/XXXXRT/Desktop/参考/不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊.wav",
    "不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊",
    i18n("中文"),
    "我在我青春韶华的时候遇到了你，还记得刚刚开学的时候，那是第一次见你，我和我朋友在楼道间打闹的时候无意间瞟到了你正在学习时的侧颜",
    i18n("中文"),
)


next(a)

timer.summary()

print("-" * 15 + "test2" + "-" * 15)
