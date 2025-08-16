"""
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
"""

import argparse
import contextlib
import logging
import os
import random
import re
from functools import partial

import gradio as gr
import psutil
import torch

from config import change_choices, custom_sort_key, get_dtype, get_weights_names, pretrained_sovits_name
from config import infer_device as default_device
from GPT_SoVITS.process_ckpt import inspect_version
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method
from GPT_SoVITS.TTS_infer_pack.TTS import NO_PROMPT_ERROR, TTS, TTS_Config
from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_high_priority():
    if os.name != "nt":
        return
    p = psutil.Process(os.getpid())
    with contextlib.suppress(psutil.AccessDenied):
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("已将进程优先级设为 High")


set_high_priority()


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inference_webui",
        description="python -s inference_webui.py zh_CN -i naive",
    )
    p.add_argument(
        "language",
        nargs="?",
        default="Auto",
        type=lang_type,
        help="Language Code, Such as zh_CN, en-US",
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


infer_ttswebui = int(args.port)
is_share = args.share

infer_device = torch.device(args.device)
device = infer_device

if torch.mps.is_available():
    device = torch.device("cpu")

dtype = get_dtype(device.index)
is_half = dtype == torch.float16

i18n = I18nAuto(language=args.language)
change_choices_gradio = partial(change_choices, i18n=i18n)

SoVITS_names, GPT_names = get_weights_names(i18n)

gpt_path = str(args.gpt) or GPT_names[0][-1]
sovits_path = str(args.sovits) or SoVITS_names[0][-1]

cnhubert_base_path = str(args.cuhubert)
bert_path = str(args.bert)

version = model_version = "v2"


dict_language_v1 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("粤语"): "all_yue",  # 全部按中文识别
    i18n("韩文"): "all_ko",  # 全部按韩文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("粤英混合"): "yue",  # 按粤英混合识别####不变
    i18n("韩英混合"): "ko",  # 按韩英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种
}
dict_language = dict_language_v1 if version == "v1" else dict_language_v2

cut_method = {
    i18n("不切"): "cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
}


path_sovits_v3 = pretrained_sovits_name["v3"]
path_sovits_v4 = pretrained_sovits_name["v4"]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
# tts_config.version = version
tts_config.update_version(version)
if gpt_path is not None:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path

print(tts_config)
tts_pipeline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path
version = tts_config.version


def inference(
    text,
    text_lang,
    ref_audio_path,
    aux_ref_audio_paths,
    prompt_text,
    prompt_lang,
    top_k,
    top_p,
    temperature,
    text_split_method,
    batch_size,
    speed_factor,
    ref_text_free,
    split_bucket,
    fragment_interval,
    seed,
    keep_random,
    parallel_infer,
    repetition_penalty,
    sample_steps,
    super_sampling,
):
    seed = -1 if keep_random else seed
    actual_seed = seed if seed not in [-1, "", None] else random.randint(0, 2**32 - 1)
    inputs = {
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": [item.name for item in aux_ref_audio_paths] if aux_ref_audio_paths is not None else [],
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size": int(batch_size),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "return_fragment": False,
        "fragment_interval": fragment_interval,
        "seed": actual_seed,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
    }
    try:
        for item in tts_pipeline.run(inputs):
            yield item, actual_seed
    except NO_PROMPT_ERROR:
        gr.Warning(i18n("V3不支持无参考文本模式，请填写参考文本！"))


v3v4set = {"v3", "v4"}


def change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
    global version, model_version, dict_language, is_lora
    model_version, version, is_lora, _, __ = inspect_version(sovits_path)
    # print(sovits_path,version, model_version, is_lora)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    if is_lora is True and is_exist is False:
        info = path_sovits + f"SoVITS {model_version}" + i18n("底模缺失，无法加载相应 LoRA 权重")
        gr.Warning(info)
        raise FileExistsError(info)
    dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = gr.skip(), gr.skip()
        else:
            prompt_text_update = gr.update(value="")
            prompt_language_update = gr.update(value=i18n("中文"))
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = gr.skip(), gr.skip()
        else:
            text_update = gr.update(value="")
            text_language_update = gr.update(value=i18n("中文"))
        if model_version in v3v4set:
            visible_sample_steps = True
            visible_inp_refs = False
        else:
            visible_sample_steps = False
            visible_inp_refs = True
        yield (
            gr.update(choices=list(dict_language.keys())),
            gr.update(choices=list(dict_language.keys())),
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
            gr.update(interactive=True if model_version not in v3v4set else False),
            gr.update(value=i18n("模型加载中，请等待"), interactive=False),
        )

    tts_pipeline.init_vits_weights(sovits_path)
    yield (
        gr.update(choices=list(dict_language.keys())),
        gr.update(choices=list(dict_language.keys())),
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
        gr.update(interactive=True if model_version not in v3v4set else False),
        gr.update(value=i18n("合成语音"), interactive=True),
    )


def change_gpt_weights(gpt_path):
    tts_pipeline.init_t2s_weights(gpt_path)


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:
    gr.HTML(
        top_html.format(
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.")
            + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        ),
        elem_classes="markdown",
    )

    with gr.Column():
        # with gr.Group():
        gr.Markdown(value=i18n("模型切换"))
        with gr.Row(equal_height=True):
            GPT_dropdown = gr.Dropdown(
                label=i18n("GPT模型列表"),
                choices=sorted(GPT_names, key=custom_sort_key),
                value=gpt_path,
                interactive=True,
            )
            SoVITS_dropdown = gr.Dropdown(
                label=i18n("SoVITS模型列表"),
                choices=sorted(SoVITS_names, key=custom_sort_key),
                value=sovits_path,
                interactive=True,
            )
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
            refresh_button.click(fn=change_choices_gradio, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])

    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown(value=i18n("*请上传并填写参考信息"))
            with gr.Row(equal_height=True):
                inp_ref = gr.Audio(
                    label=i18n("主参考音频(请上传3~10秒内参考音频，超过会报错！)"),
                    type="filepath",
                    waveform_options={"show_recording_waveform": False},
                )
                inp_refs = gr.File(
                    label=i18n("辅参考音频(可选多个，或不选)"),
                    file_count="multiple",
                    visible=True if model_version != "v3" else False,
                )
            prompt_text = gr.Textbox(label=i18n("主参考音频的文本"), value="", lines=2)
            with gr.Row(equal_height=True):
                prompt_language = gr.Dropdown(
                    label=i18n("主参考音频的语种"), choices=list(dict_language.keys()), value=i18n("中文")
                )
                with gr.Column():
                    ref_text_free = gr.Checkbox(
                        label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"),
                        value=False,
                        interactive=True if model_version != "v3" else False,
                        show_label=True,
                    )
                    gr.Markdown(
                        i18n("使用无参考文本模式时建议使用微调的GPT")
                        + "<br>"
                        + i18n("听不清参考音频说的啥(不晓得写啥)可以开。开启后无视填写的参考文本。")
                    )

        with gr.Column():
            gr.Markdown(value=i18n("*请填写需要合成的目标文本和语种模式"))
            text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=20, max_lines=20)
            text_language = gr.Dropdown(
                label=i18n("需要合成的文本的语种"), choices=list(dict_language.keys()), value=i18n("中文")
            )

    with gr.Group():
        gr.Markdown(value=i18n("推理设置"))
        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Row(equal_height=True):
                    batch_size = gr.Slider(
                        minimum=1, maximum=200, step=1, label=i18n("batch_size"), value=20, interactive=True
                    )
                    sample_steps = gr.Radio(
                        label=i18n("采样步数(仅对V3/4生效)"), value=32, choices=[4, 8, 16, 32, 64, 128], visible=True
                    )
                with gr.Row(equal_height=True):
                    fragment_interval = gr.Slider(
                        minimum=0.01, maximum=1, step=0.01, label=i18n("分段间隔(秒)"), value=0.3, interactive=True
                    )
                    speed_factor = gr.Slider(
                        minimum=0.6, maximum=1.65, step=0.05, label="语速", value=1.0, interactive=True
                    )
                with gr.Row(equal_height=True):
                    top_k = gr.Slider(minimum=1, maximum=100, step=1, label=i18n("top_k"), value=5, interactive=True)
                    top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n("top_p"), value=1, interactive=True)
                with gr.Row(equal_height=True):
                    temperature = gr.Slider(
                        minimum=0, maximum=1, step=0.05, label=i18n("temperature"), value=1, interactive=True
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0, maximum=2, step=0.05, label=i18n("重复惩罚"), value=1.35, interactive=True
                    )

            with gr.Column():
                with gr.Row(equal_height=True):
                    how_to_cut = gr.Dropdown(
                        label=i18n("怎么切"),
                        choices=[
                            i18n("不切"),
                            i18n("凑四句一切"),
                            i18n("凑50字一切"),
                            i18n("按中文句号。切"),
                            i18n("按英文句号.切"),
                            i18n("按标点符号切"),
                        ],
                        value=i18n("凑四句一切"),
                        interactive=True,
                        scale=1,
                    )
                    super_sampling = gr.Checkbox(
                        label=i18n("音频超采样(仅对V3生效))"), value=False, interactive=True, show_label=True
                    )

                with gr.Row(equal_height=True):
                    parallel_infer = gr.Checkbox(label=i18n("并行推理"), value=True, interactive=True, show_label=True)
                    split_bucket = gr.Checkbox(
                        label=i18n("数据分桶(并行推理时会降低一点计算量)"),
                        value=True,
                        interactive=True,
                        show_label=True,
                    )

                with gr.Row(equal_height=True):
                    seed = gr.Number(label=i18n("随机种子"), value=-1)
                    keep_random = gr.Checkbox(label=i18n("保持随机"), value=True, interactive=True, show_label=True)

                output = gr.Audio(
                    label=i18n("输出的语音"),
                    waveform_options={"show_recording_waveform": False},
                )
                with gr.Row(equal_height=True):
                    inference_button = gr.Button(i18n("合成语音"), variant="primary")
                    stop_infer = gr.Button(i18n("终止合成"), variant="primary")

        inference_button.click(
            inference,
            [
                text,
                text_language,
                inp_ref,
                inp_refs,
                prompt_text,
                prompt_language,
                top_k,
                top_p,
                temperature,
                how_to_cut,
                batch_size,
                speed_factor,
                ref_text_free,
                split_bucket,
                fragment_interval,
                seed,
                keep_random,
                parallel_infer,
                repetition_penalty,
                sample_steps,
                super_sampling,
            ],
            [output, seed],
        )
        stop_infer.click(tts_pipeline.stop, [], [])
        SoVITS_dropdown.change(
            change_sovits_weights,
            [SoVITS_dropdown, prompt_language, text_language],
            [
                prompt_language,
                text_language,
                prompt_text,
                prompt_language,
                text,
                text_language,
                sample_steps,
                inp_refs,
                ref_text_free,
                inference_button,
            ],
        )  #
        GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

    with gr.Group():
        gr.Markdown(
            value=i18n(
                "文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。合成会根据文本的换行分开合成再拼起来。"
            )
        )
        with gr.Row(equal_height=True):
            text_inp = gr.Textbox(label=i18n("需要合成的切分前文本"), value="", lines=4)
            with gr.Column():
                _how_to_cut = gr.Radio(
                    label=i18n("怎么切"),
                    choices=[
                        i18n("不切"),
                        i18n("凑四句一切"),
                        i18n("凑50字一切"),
                        i18n("按中文句号。切"),
                        i18n("按英文句号.切"),
                        i18n("按标点符号切"),
                    ],
                    value=i18n("凑四句一切"),
                    interactive=True,
                )
                cut_text = gr.Button(i18n("切分"), variant="primary")

            def to_cut(text_inp, how_to_cut):
                if len(text_inp.strip()) == 0 or text_inp == []:
                    return ""
                method = get_method(cut_method[how_to_cut])
                return method(text_inp)

            text_opt = gr.Textbox(label=i18n("切分后文本"), value="", lines=4)
            cut_text.click(to_cut, [text_inp, _how_to_cut], [text_opt])
        gr.Markdown(value=i18n("后续将支持转音素、手工修改音素、语音合成分步执行。"))

if __name__ == "__main__":
    app.queue().launch(  # concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=infer_ttswebui,
        # quiet=True,
    )
