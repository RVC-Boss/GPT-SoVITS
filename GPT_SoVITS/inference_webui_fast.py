# -*- coding: utf-8 -*-
"""
GPT-SoVITS WebUI 精简版
保留功能：模型持久化、参考音频持久化、推理参数持久化、记住最后选中预设
核心优化：抽离持久化逻辑到 persistence_tools.py，主文件大幅精简，结构清晰
"""
import psutil
import os
import sys
import json
import yaml
import random
import re
import shutil
from pathlib import Path

import torch
import gradio as gr

# 设置进程优先级（仅Windows有效）
def set_high_priority():
    if os.name != "nt":
        return
    p = psutil.Process(os.getpid())
    try:
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("已将进程优先级设为 High")
    except psutil.AccessDenied:
        print("权限不足，无法修改优先级（请用管理员运行）")
set_high_priority()

# ===================== 导入自定义持久化工具类（核心精简关键） =====================
from persistence_tools import (
    init_last_selected_models, read_last_selected_models, write_last_selected_models,
    read_last_selected_preset, write_last_selected_preset, clear_last_selected_preset,
    load_ref_presets, get_preset_by_name, save_ref_preset_core, delete_ref_preset_core,
    load_infer_settings, save_infer_settings_core, restore_default_infer_settings_core,
    REF_AUDIO_DIR
)

# ===================== 原有核心依赖导入 =====================
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

# 屏蔽无关日志
import logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

# 配置参数初始化
infer_ttswebui = int(os.environ.get("infer_ttswebui", 9872))
is_share = eval(os.environ.get("is_share", "False"))
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)
version = model_version = os.environ.get("version", "v2")

# 标记是否直接打开推理页
is_direct_launch = (gpt_path is None) and (sovits_path is None)

# 多语言配置
from tools.i18n.i18n import I18nAuto, scan_language_list
language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

# TTS 推理核心
from TTS_infer_pack.text_segmentation_method import get_method
from TTS_infer_pack.TTS import NO_PROMPT_ERROR, TTS, TTS_Config

# 样式与配置工具
from tools.assets import css, js, top_html
from config import change_choices, get_weights_names, name2gpt_path, name2sovits_path
from process_ckpt import get_sovits_version_from_path_fast

# ===================== 全局变量初始化 =====================
# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 语种字典
dict_language_v1 = {
    i18n("中文"): "all_zh",
    i18n("英文"): "en",
    i18n("日文"): "all_ja",
    i18n("中英混合"): "zh",
    i18n("日英混合"): "ja",
    i18n("多语种混合"): "auto",
}
dict_language_v2 = {
    i18n("中文"): "all_zh",
    i18n("英文"): "en",
    i18n("日文"): "all_ja",
    i18n("粤语"): "all_yue",
    i18n("韩文"): "all_ko",
    i18n("中英混合"): "zh",
    i18n("日英混合"): "ja",
    i18n("粤英混合"): "yue",
    i18n("韩英混合"): "ko",
    i18n("多语种混合"): "auto",
    i18n("多语种混合(粤语)"): "auto_yue",
}
dict_language = dict_language_v1 if version == "v1" else dict_language_v2

# 文本切分方法
cut_method = {
    i18n("不切"): "cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
}

# V3/V4 标记
v3v4set = {"v3", "v4"}

# 模型列表初始化
SoVITS_names, GPT_names = get_weights_names()
from config import pretrained_sovits_name
path_sovits_v3 = pretrained_sovits_name["v3"]
path_sovits_v4 = pretrained_sovits_name["v4"]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)

# TTS 配置与管道初始化
tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
tts_config.update_version(version)
if gpt_path is not None and "！" not in gpt_path and "!" not in gpt_path:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None and "！" not in sovits_path and "!" not in sovits_path:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path

tts_pipeline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path
version = tts_config.version

# 参考预设全局变量
ref_presets = load_ref_presets()
preset_names = [p["name"] for p in ref_presets] if ref_presets else []

# ===================== 精简版辅助函数 =====================
def custom_sort_key(s):
    """自定义排序键（数字自然排序）"""
    parts = re.split("(\d+)", s)
    return [int(part) if part.isdigit() else part for part in parts]

def init_ui_preset_config():
    """初始化UI配置（优先加载最后选中的预设）"""
    global ref_presets, preset_names
    ref_presets = load_ref_presets()
    preset_names = [p["name"] for p in ref_presets] if ref_presets else []
    is_interactive = bool(preset_names)
    
    # 优先读取最后选中的预设
    last_selected = read_last_selected_preset()
    default_selected = last_selected if (last_selected and last_selected in preset_names) else (preset_names[0] if preset_names else None)
    default_preset = get_preset_by_name(default_selected)
    
    return (
        gr.update(choices=preset_names, value=default_selected, interactive=is_interactive),
        default_preset["name"],
        default_preset["ref_audio_path"],
        default_preset["prompt_text"],
        default_preset["prompt_language"]
    )

def update_popup_text(preset_name, is_delete):
    """更新弹窗提示文本"""
    preset_name = preset_name.strip()
    if is_delete:
        return gr.update(value=i18n(f"确定要删除配置「{preset_name}」吗？删除后无法恢复！"))
    else:
        return gr.update(value=i18n(f"配置「{preset_name}」已存在，确定要覆盖吗？覆盖后无法恢复！"))

def reset_confirm_result():
    """重置确认结果为False"""
    return False

def save_ref_preset_wrapper(preset_name, ref_audio_path, prompt_text, prompt_language, confirm_override=False):
    """保存预设包装器（适配Gradio输出）"""
    msg, success, new_preset_names = save_ref_preset_core(preset_name, ref_audio_path, prompt_text, prompt_language, confirm_override)
    style = gr.update(elem_classes=["config-error-border"]) if not success else gr.update(elem_classes=["config-default-border"])
    dropdown_update = gr.update(choices=new_preset_names, value=preset_name if success and preset_name in new_preset_names else (new_preset_names[0] if new_preset_names else None), interactive=bool(new_preset_names))
    return msg, style, dropdown_update

def delete_ref_preset_wrapper(preset_name):
    """删除预设包装器（适配Gradio输出）"""
    msg, new_preset_names, new_selected = delete_ref_preset_core(preset_name)
    dropdown_update = gr.update(choices=new_preset_names, value=new_selected, interactive=bool(new_preset_names))
    new_preset = get_preset_by_name(new_selected)
    
    return (
        msg,
        dropdown_update,
        new_preset["name"],
        new_preset["ref_audio_path"],
        new_preset["prompt_text"],
        new_preset["prompt_language"]
    )

def on_preset_selected(preset_name):
    """预设切换回调（记录最后选中）"""
    if not preset_name or not ref_presets:
        return "", None, "", i18n("中文")
    preset = get_preset_by_name(preset_name)
    audio_path = preset["ref_audio_path"]
    
    # 记录最后选中的预设
    write_last_selected_preset(preset_name)
    
    return preset["name"], audio_path, preset["prompt_text"], preset["prompt_language"]

# ===================== 推理核心函数 =====================
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
    """语音合成推理核心"""
    seed = -1 if keep_random else seed
    actual_seed = seed if seed not in [-1, "", None] else random.randint(0, 2**32 - 1)
    ref_audio_path = ref_audio_path if ref_audio_path else ""
    
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
        return i18n("V3不支持无参考文本模式，请填写参考文本！")

# ===================== 模型切换函数（保留预设记忆） =====================
def change_sovits_weights(sovits_path, current_gpt_selected, prompt_language=None, text_language=None):
    """切换SoVITS模型（保留最后选中预设）"""
    if "！" in sovits_path or "!" in sovits_path:
        sovits_path = name2sovits_path[sovits_path]
    
    global version, model_version, dict_language, if_lora_v3, ref_presets, preset_names
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    
    # 底模缺失校验
    if if_lora_v3 and not is_exist:
        info = path_sovits + "SoVITS %s" % model_version + i18n("底模缺失，无法加载相应 LoRA 权重")
        return i18n(info)
    
    # 更新语种字典与预设
    dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    ref_presets = load_ref_presets()
    preset_names = [p["name"] for p in ref_presets] if ref_presets else []
    
    # 恢复最后选中的预设
    last_selected = read_last_selected_preset()
    default_selected = last_selected if (last_selected and last_selected in preset_names) else (preset_names[0] if preset_names else None)
    
    # 更新模型配置文件
    if is_direct_launch:
        valid_gpt_path = current_gpt_selected if (current_gpt_selected in GPT_names and os.path.exists(current_gpt_selected)) else GPT_names[-1]
        write_last_selected_models(valid_gpt_path, sovits_path, version)
    
    # 初始化返回值
    if prompt_language is None or text_language is None:
        return
    
    # 语种兼容性校验
    prompt_text_update, prompt_language_update = {"__type__": "update"}, {"__type__": "update", "value": prompt_language}
    if prompt_language not in list(dict_language.keys()):
        prompt_text_update = {"__type__": "update", "value": ""}
        prompt_language_update = {"__type__": "update", "value": i18n("中文")}
    
    text_update, text_language_update = {"__type__": "update"}, {"__type__": "update", "value": text_language}
    if text_language not in list(dict_language.keys()):
        text_update = {"__type__": "update", "value": ""}
        text_language_update = {"__type__": "update", "value": i18n("中文")}
    
    # V3/V4 特殊配置
    visible_sample_steps = model_version in v3v4set
    visible_inp_refs = not visible_sample_steps
    ref_text_free_interactive = model_version not in v3v4set
    
    # 加载中状态
    yield (
        {"__type__": "update", "choices": list(dict_language.keys())},
        {"__type__": "update", "choices": list(dict_language.keys())},
        prompt_text_update,
        prompt_language_update,
        text_update,
        text_language_update,
        {"__type__": "update", "interactive": visible_sample_steps, "value": 32},
        {"__type__": "update", "visible": visible_inp_refs},
        {"__type__": "update", "interactive": ref_text_free_interactive},
        {"__type__": "update", "value": i18n("模型加载中，请等待"), "interactive": False},
        gr.update(choices=preset_names, value=default_selected, interactive=bool(preset_names)),
    )
    
    # 加载模型权重
    tts_pipeline.init_vits_weights(sovits_path)
    
    # 加载完成状态
    yield (
        {"__type__": "update", "choices": list(dict_language.keys())},
        {"__type__": "update", "choices": list(dict_language.keys())},
        prompt_text_update,
        prompt_language_update,
        text_update,
        text_language_update,
        {"__type__": "update", "interactive": visible_sample_steps, "value": 32},
        {"__type__": "update", "visible": visible_inp_refs},
        {"__type__": "update", "interactive": ref_text_free_interactive},
        {"__type__": "update", "value": i18n("合成语音"), "interactive": True},
        gr.update(choices=preset_names, value=default_selected, interactive=bool(preset_names)),
    )
    
    # 更新 weight.json
    with open("./weight.json", "r") as f:
        data = json.loads(f.read())
    data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as f:
        json.dump(data, f)

def change_gpt_weights(gpt_path):
    """切换GPT模型"""
    if "！" in gpt_path or "!" in gpt_path:
        gpt_path = name2gpt_path[gpt_path]
    tts_pipeline.init_t2s_weights(gpt_path)
    
    if is_direct_launch:
        current_sovits_path = sovits_path if 'sovits_path' in globals() else SoVITS_names[0]
        current_version = version if 'version' in globals() else "v2"
        write_last_selected_models(gpt_path, current_sovits_path, current_version)

# ===================== 推理参数持久化包装函数 =====================
def init_infer_settings():
    """初始化推理参数"""
    settings = load_infer_settings()
    return (
        settings["batch_size"],
        settings["sample_steps"],
        settings["fragment_interval"],
        settings["speed_factor"],
        settings["top_k"],
        settings["top_p"],
        settings["temperature"],
        settings["repetition_penalty"],
        settings["how_to_cut"],
        settings["super_sampling"],
        settings["parallel_infer"],
        settings["split_bucket"],
        settings["seed"],
        settings["keep_random"]
    )

def save_infer_settings_wrapper(batch_size, sample_steps, fragment_interval, speed_factor,
                                top_k, top_p, temperature, repetition_penalty, how_to_cut,
                                super_sampling, parallel_infer, split_bucket, seed, keep_random):
    """保存推理参数包装器"""
    settings = {
        "batch_size": batch_size,
        "sample_steps": sample_steps,
        "fragment_interval": fragment_interval,
        "speed_factor": speed_factor,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "how_to_cut": how_to_cut,
        "super_sampling": super_sampling,
        "parallel_infer": parallel_infer,
        "split_bucket": split_bucket,
        "seed": -1 if keep_random else seed,
        "keep_random": keep_random
    }
    return save_infer_settings_core(settings)

# ===================== Gradio UI 构建（精简版） =====================
custom_css = """
/* 保存失败红边框样式（强制覆盖默认样式） */
.config-error-border {
    border: 2px solid #ff3b30 !important;
    border-radius: 4px !important;
    padding: 6px !important;
}
/* 保存成功/默认边框样式（还原原有样式） */
.config-default-border {
    border: 1px solid #e5e7eb !important;
    border-radius: 4px !important;
    padding: 6px !important;
}
/* 模拟弹窗样式：紧凑居中+自动换行+窄宽度 */
.simulated-popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    padding: 15px;
    z-index: 9999;
    min-width: 300px;
    max-width: 350px;
    text-align: center;
}
/* 弹窗文本自动换行 */
.simulated-popup .markdown-text {
    word-wrap: break-word;
    line-height: 1.5;
    color: #333;
}
/* 弹窗遮罩层 */
.popup-mask {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.4);
    z-index: 9998;
}
/* 按钮区域：紧凑间距+适配宽度 */
.popup-buttons {
    margin-top: 15px;
    display: flex;
    justify-content: center;
    gap: 10px;
}
/* 按钮样式优化：适配弹窗宽度 */
.popup-buttons button {
    min-width: 80px;
    padding: 6px 12px;
}
.btn-group-spacing {
    display: flex !important;
    flex-direction: row !important;
    gap: 10px !important;
    padding:5px 10px 5px 10px;
    align-items: center !important;
    justify-content: flex-start !important;
}
.btn-group-spacing button{
    border-radius:5px;
}
"""

with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css + custom_css) as app:
    # 状态变量
    current_preset = gr.State("")
    confirm_flag = gr.State(False)
    infer_confirm_flag = gr.State(False)
    infer_restore_flag = gr.State(False)
    
    # 顶部HTML
    gr.HTML(top_html.format(
        i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.")
        + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
    ), elem_classes="markdown")

    # 模型切换区域
    with gr.Column():
        gr.Markdown(value=i18n("模型切换"))
        with gr.Row():
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
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])

    # 核心功能区域
    with gr.Row():
        with gr.Column():
            gr.Markdown(value=i18n("*请上传并填写参考信息（支持多组预设配置切换/删除，主参考音频为必填项，自动持久化不丢失）"))
            
            # 预设选择与名称输入
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label=i18n("选择预设参考配置"),
                    choices=preset_names,
                    value=preset_names[0] if preset_names else None,
                    interactive=bool(preset_names),
                )
                preset_name_input = gr.Textbox(
                    label=i18n("当前配置名称（保存时使用）"),
                    value="",
                    placeholder=i18n("填写配置名称，点击保存按钮即可持久化"),
                    lines=1,
                    scale=1
                )
            
            # 配置提示框
            save_config_msg = gr.Textbox(
                label=i18n("配置操作提示"),
                value="",
                lines=1,
                interactive=False,
                elem_classes=["config-default-border"]
            )
            
            # 保存/删除按钮
            with gr.Row():
                save_ref_config_btn = gr.Button(i18n("保存当前参考为选中配置"), variant="primary")
                delete_ref_config_btn = gr.Button(i18n("删除当前选中配置"), variant="primary")
            
            # 音频与文本输入
            with gr.Row():
                inp_ref = gr.Audio(
                    label=i18n("主参考音频(3~10秒)【必填，自动持久化】"),
                    type="filepath",
                    value=None
                )
                inp_refs = gr.File(
                    label=i18n("辅参考音频(可选多个)"),
                    file_count="multiple",
                    visible=True if model_version != "v3" else False,
                )
            
            prompt_text = gr.Textbox(
                label=i18n("主参考音频的文本【可选】"),
                value="",
                lines=2
            )
            
            with gr.Row():
                prompt_language = gr.Dropdown(
                    label=i18n("主参考音频的语种"),
                    choices=list(dict_language.keys()),
                    value=i18n("中文")
                )
                ref_text_free = gr.Checkbox(
                    label=i18n("开启无参考文本模式"),
                    value=False,
                    interactive=True if model_version != "v3" else False,
                    show_label=True,
                )
            
            # 模拟弹窗（删除/覆盖）
            delete_mask = gr.Column(visible=False, elem_classes=["popup-mask"])
            with gr.Column(visible=False, elem_classes=["simulated-popup"]) as delete_popup:
                delete_text = gr.Markdown(value="")
                with gr.Row(elem_classes=["popup-buttons"]):
                    delete_confirm_btn = gr.Button(i18n("确认删除"), variant="primary")
                    delete_cancel_btn = gr.Button(i18n("取消"), variant="primary")
            
            override_mask = gr.Column(visible=False, elem_classes=["popup-mask"])
            with gr.Column(visible=False, elem_classes=["simulated-popup"]) as override_popup:
                override_text = gr.Markdown(value="")
                with gr.Row(elem_classes=["popup-buttons"]):
                    override_confirm_btn = gr.Button(i18n("确认覆盖"), variant="primary")
                    override_cancel_btn = gr.Button(i18n("取消"), variant="primary")
            
            # 事件绑定：预设切换
            preset_dropdown.change(
                fn=on_preset_selected,
                inputs=[preset_dropdown],
                outputs=[preset_name_input, inp_ref, prompt_text, prompt_language]
            )
            
            # 事件绑定：保存预设
            save_ref_config_btn.click(
                fn=lambda pname: (pname.strip(), any(p["name"].strip() == pname.strip() for p in ref_presets)),
                inputs=[preset_name_input],
                outputs=[current_preset, confirm_flag]
            ).then(
                fn=lambda exists, pname: (
                    gr.update(visible=exists),
                    gr.update(visible=exists),
                    update_popup_text(pname, False)
                ),
                inputs=[confirm_flag, preset_name_input],
                outputs=[override_mask, override_popup, override_text]
            ).then(
                fn=save_ref_preset_wrapper,
                inputs=[preset_name_input, inp_ref, prompt_text, prompt_language, confirm_flag],
                outputs=[save_config_msg, save_config_msg, preset_dropdown]
            )
            
            # 事件绑定：覆盖确认
            override_confirm_btn.click(
                fn=lambda: (True, gr.update(visible=False), gr.update(visible=False)),
                inputs=[],
                outputs=[confirm_flag, override_mask, override_popup]
            ).then(
                fn=save_ref_preset_wrapper,
                inputs=[preset_name_input, inp_ref, prompt_text, prompt_language, confirm_flag],
                outputs=[save_config_msg, save_config_msg, preset_dropdown]
            ).then(
                fn=reset_confirm_result,
                inputs=[],
                outputs=[confirm_flag]
            )
            
            # 事件绑定：覆盖取消
            override_cancel_btn.click(
                fn=lambda: (False, gr.update(visible=False), gr.update(visible=False), i18n("覆盖操作已取消")),
                inputs=[],
                outputs=[confirm_flag, override_mask, override_popup, save_config_msg]
            )
            
            # 事件绑定：删除预设
            delete_ref_config_btn.click(
                fn=lambda pname: (pname, gr.update(visible=True), gr.update(visible=True), update_popup_text(pname, True)),
                inputs=[preset_dropdown],
                outputs=[current_preset, delete_mask, delete_popup, delete_text]
            )
            
            # 事件绑定：删除确认
            delete_confirm_btn.click(
                fn=lambda: (True, gr.update(visible=False), gr.update(visible=False)),
                inputs=[],
                outputs=[confirm_flag, delete_mask, delete_popup]
            ).then(
                fn=delete_ref_preset_wrapper,
                inputs=[preset_dropdown],
                outputs=[save_config_msg, preset_dropdown, preset_name_input, inp_ref, prompt_text, prompt_language]
            ).then(
                fn=reset_confirm_result,
                inputs=[],
                outputs=[confirm_flag]
            )
            
            # 事件绑定：删除取消
            delete_cancel_btn.click(
                fn=lambda pname: (False, gr.update(visible=False), gr.update(visible=False), i18n("删除操作已取消")),
                inputs=[preset_dropdown],
                outputs=[confirm_flag, delete_mask, delete_popup, save_config_msg]
            )

        with gr.Column():
            gr.Markdown(value=i18n("*请填写需要合成的目标文本和语种模式"))
            
            # 操作教程提示
            gr.HTML('''
            <div style="height: auto; min-height: 240px; width: 100%; padding: 16px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: #fafafa;">
                <h4 style="margin: 0 0 12px 0; color: #1f2937; font-size: 16px;">GPT-SoVITS 新增功能说明</h4>
                 <div style="margin-bottom: 10px; line-height: 1.6; font-size: 13px; color: #374151;">
                    <strong style="color: #d97706;">1.默认模型</strong>：进入推理页面时，自动加载最后一次使用的模型。
                </div>
                <div style="margin-bottom: 10px; line-height: 1.6; font-size: 13px; color: #374151;">
                    <strong style="color: #d97706;">2.参考音频</strong>：自动保存到 GPT_SoVITS/ref_audios/，多组预设可切换，重启不丢失。
                </div>
                <div style="margin-bottom: 10px; line-height: 1.6; font-size: 13px; color: #374151;">
                    <strong style="color: #d97706;">3.推理参数</strong>：可保存自定义参数，恢复默认值，无需每次重新调整。
                </div>
            </div>
            ''')
            
            # 目标文本输入
            text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=20, max_lines=20)
            text_language = gr.Dropdown(
                label=i18n("需要合成的文本的语种"),
                choices=list(dict_language.keys()),
                value=i18n("中文")
            )

    # 推理设置区域
    with gr.Group():
        gr.Markdown(value=i18n("推理设置"))
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    batch_size = gr.Slider(minimum=1, maximum=200, step=1, label=i18n("batch_size"), value=20, interactive=True)
                    sample_steps = gr.Radio(label=i18n("采样步数(仅V3/4生效)"), value=32, choices=[4, 8, 16, 32, 64, 128], visible=True)
                with gr.Row():
                    fragment_interval = gr.Slider(minimum=0.01, maximum=1, step=0.01, label=i18n("分段间隔(秒)"), value=0.2, interactive=True)
                    speed_factor = gr.Slider(minimum=0.6, maximum=1.65, step=0.05, label="语速", value=1.0, interactive=True)
                with gr.Row():
                    top_k = gr.Slider(minimum=1, maximum=100, step=1, label=i18n("top_k"), value=5, interactive=True)
                    top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n("top_p"), value=1, interactive=True)
                with gr.Row():
                    temperature = gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n("temperature"), value=1, interactive=True)
                    repetition_penalty = gr.Slider(minimum=0, maximum=2, step=0.05, label=i18n("重复惩罚"), value=1.35, interactive=True)

            with gr.Column():
                with gr.Row():
                    how_to_cut = gr.Dropdown(
                        label=i18n("怎么切"),
                        choices=list(cut_method.keys()),
                        value=i18n("凑四句一切"),
                        interactive=True,
                        scale=1,
                    )
                    super_sampling = gr.Checkbox(label=i18n("音频超采样(仅V3生效)"), value=False, interactive=True, show_label=True)

                with gr.Row():
                    parallel_infer = gr.Checkbox(label=i18n("并行推理"), value=True, interactive=True, show_label=True)
                    split_bucket = gr.Checkbox(label=i18n("数据分桶"), value=True, interactive=True, show_label=True)

                with gr.Row():
                    seed = gr.Number(label=i18n("随机种子"), value=-1)
                    keep_random = gr.Checkbox(label=i18n("保持随机"), value=True, interactive=True, show_label=True)

                output = gr.Audio(label=i18n("输出的语音"))
        
        # 推理操作按钮
        with gr.Row():         
                 # ========== 新增：两个按钮（保存推理设置、恢复默认设置） ==========
            with gr.Row(elem_classes=["btn-group-spacing"]):
                 save_infer_settings_btn = gr.Button(i18n("保存推理设置"), variant="primary")
                 restore_default_settings_btn = gr.Button(i18n("恢复默认设置"), variant="primary")
            with gr.Row(elem_classes=["btn-group-spacing"]):
                 inference_button = gr.Button(i18n("合成语音"), variant="primary")
                 stop_infer = gr.Button(i18n("终止合成"), variant="primary")

        # ===================== 推理保存确认弹窗（原有，保持不变） =====================
        infer_save_mask = gr.Column(visible=False, elem_classes=["popup-mask"])
        with gr.Column(visible=False, elem_classes=["simulated-popup"]) as infer_save_popup:
            infer_save_text = gr.Markdown(value=i18n("确定要保存当前推理设置吗？保存后将覆盖原有配置！"))
            with gr.Row(elem_classes=["popup-buttons"]):
                infer_save_confirm_btn = gr.Button(i18n("确认保存"), variant="primary")
                infer_save_cancel_btn = gr.Button(i18n("取消"), variant="primary")

        # ===================== 【新增】：推理恢复默认确认弹窗 =====================
        infer_restore_mask = gr.Column(visible=False, elem_classes=["popup-mask"])
        with gr.Column(visible=False, elem_classes=["simulated-popup"]) as infer_restore_popup:
            infer_restore_text = gr.Markdown(value=i18n("确定要恢复推理设置为默认值吗？所有自定义参数将被覆盖，且无法恢复！"))
            with gr.Row(elem_classes=["popup-buttons"]):
                infer_restore_confirm_btn = gr.Button(i18n("确认恢复"), variant="primary")
                infer_restore_cancel_btn = gr.Button(i18n("取消"), variant="primary")

        # 推理保存专用提示框（原有，保持不变）
        infer_save_msg = gr.Textbox(
            label=i18n("推理设置操作提示"),
            value="",
            lines=1,
            interactive=False,
            elem_classes=["config-default-border"]
        )

        # 事件绑定：保存推理参数
        save_infer_settings_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
            inputs=[],
            outputs=[infer_save_mask, infer_save_popup]
        )

        infer_save_confirm_btn.click(
            fn=lambda: (True, gr.update(visible=False), gr.update(visible=False)),
            inputs=[],
            outputs=[infer_confirm_flag, infer_save_mask, infer_save_popup]
        ).then(
            fn=save_infer_settings_wrapper,
            inputs=[batch_size, sample_steps, fragment_interval, speed_factor,
                    top_k, top_p, temperature, repetition_penalty, how_to_cut,
                    super_sampling, parallel_infer, split_bucket, seed, keep_random],
            outputs=[infer_save_msg]
        ).then(
            fn=lambda: False,
            inputs=[],
            outputs=[infer_confirm_flag]
        )

        infer_save_cancel_btn.click(
            fn=lambda: (False, gr.update(visible=False), gr.update(visible=False), i18n("保存操作已取消")),
            inputs=[],
            outputs=[infer_confirm_flag, infer_save_mask, infer_save_popup, infer_save_msg]
        )

        # 事件绑定：恢复默认推理参数
        restore_default_settings_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
            inputs=[],
            outputs=[infer_restore_mask, infer_restore_popup]
        )

        infer_restore_confirm_btn.click(
            fn=lambda: (True, gr.update(visible=False), gr.update(visible=False)),
            inputs=[],
            outputs=[infer_restore_flag, infer_restore_mask, infer_restore_popup]
        ).then(
            fn=restore_default_infer_settings_core,
            inputs=[],
            outputs=[batch_size, sample_steps, fragment_interval, speed_factor,
                    top_k, top_p, temperature, repetition_penalty, how_to_cut,
                    super_sampling, parallel_infer, split_bucket, seed, keep_random]
        ).then(
            fn=lambda: i18n("推理设置已恢复为默认值！"),
            inputs=[],
            outputs=[infer_save_msg]
        ).then(
            fn=lambda: False,
            inputs=[],
            outputs=[infer_restore_flag]
        )

        infer_restore_cancel_btn.click(
            fn=lambda: (False, gr.update(visible=False), gr.update(visible=False), i18n("恢复操作已取消")),
            inputs=[],
            outputs=[infer_restore_flag, infer_restore_mask, infer_restore_popup, infer_save_msg]
        )

        # 事件绑定：合成语音
        inference_button.click(
            inference,
            inputs=[text, text_language, inp_ref, inp_refs, prompt_text, prompt_language,
                    top_k, top_p, temperature, how_to_cut, batch_size, speed_factor,
                    ref_text_free, split_bucket, fragment_interval, seed, keep_random,
                    parallel_infer, repetition_penalty, sample_steps, super_sampling],
            outputs=[output, seed]
        )

        # 事件绑定：终止合成
        stop_infer.click(tts_pipeline.stop, [], [])

        # 事件绑定：SoVITS模型切换
        SoVITS_dropdown.change(
            change_sovits_weights,
            inputs=[SoVITS_dropdown, GPT_dropdown, prompt_language, text_language],
            outputs=[prompt_language, text_language, prompt_text, prompt_language,
                    text, text_language, sample_steps, inp_refs, ref_text_free,
                    inference_button, preset_dropdown]
        )

        # 事件绑定：GPT模型切换
        GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

    # 文本切分工具
    with gr.Group():
        gr.Markdown(value=i18n("文本切分工具（太长的文本建议先切分）"))
        with gr.Row():
            text_inp = gr.Textbox(label=i18n("切分前文本"), value="", lines=4)
            _how_to_cut = gr.Radio(label=i18n("怎么切"), choices=list(cut_method.keys()), value=i18n("凑四句一切"), interactive=True)
            cut_text = gr.Button(i18n("切分"), variant="primary")
            text_opt = gr.Textbox(label=i18n("切分后文本"), value="", lines=4)

            def to_cut(text_inp, how_to_cut):
                if not text_inp.strip():
                    return ""
                method = get_method(cut_method[how_to_cut])
                return method(text_inp)

            cut_text.click(to_cut, [text_inp, _how_to_cut], [text_opt])

    # 页面加载初始化
    app.load(fn=init_ui_preset_config, inputs=[], outputs=[preset_dropdown, preset_name_input, inp_ref, prompt_text, prompt_language])
    app.load(fn=init_infer_settings, inputs=[], outputs=[batch_size, sample_steps, fragment_interval, speed_factor,
                    top_k, top_p, temperature, repetition_penalty, how_to_cut,
                    super_sampling, parallel_infer, split_bucket, seed, keep_random])

# ===================== 应用入口 =====================
if __name__ == "__main__":
    # 初始化 weight.json
    if not os.path.exists("./weight.json"):
        with open("./weight.json", "w", encoding="utf-8") as file:
            json.dump({"GPT": {}, "SoVITS": {}}, file)
    
    # 直接打开推理页的模型配置初始化
    if is_direct_launch:
        default_gpt_path = gpt_path
        default_sovits_path = sovits_path
        default_version = version
        
        last_selected_data = read_last_selected_models()
        if last_selected_data is None:
            init_last_selected_models(default_gpt_path, default_sovits_path, default_version)
        else:
            # 加载保存的模型配置
            saved_gpt = last_selected_data["gpt_model_path"]
            saved_sovits = last_selected_data["sovits_model_path"]
            saved_version = last_selected_data["version"]
            
            if saved_gpt in GPT_names and os.path.exists(saved_gpt):
                gpt_path = saved_gpt
            if saved_sovits in SoVITS_names and os.path.exists(saved_sovits):
                sovits_path = saved_sovits
            if saved_version in ["v1", "v2", "v3", "v4"]:
                version = saved_version
        
        # 主动加载模型权重
        valid_gpt_path = gpt_path if (gpt_path in GPT_names and os.path.exists(gpt_path)) else GPT_names[-1]
        if "！" in valid_gpt_path or "!" in valid_gpt_path:
            valid_gpt_path = name2gpt_path[valid_gpt_path]
        tts_pipeline.init_t2s_weights(valid_gpt_path)
        
        valid_sovits_path = sovits_path if (sovits_path in SoVITS_names and os.path.exists(sovits_path)) else SoVITS_names[0]
        if "！" in valid_sovits_path or "!" in valid_sovits_path:
            valid_sovits_path = name2sovits_path[valid_sovits_path]
        tts_pipeline.init_vits_weights(valid_sovits_path)
        
        # 更新 weight.json
        with open("./weight.json", "r", encoding="utf-8") as f:
            weight_data = json.loads(f.read())
        weight_data["GPT"][version] = valid_gpt_path
        weight_data["SoVITS"][version] = valid_sovits_path
        with open("./weight.json", "w", encoding="utf-8") as f:
            json.dump(weight_data, f, ensure_ascii=False, indent=4)
    
    # 启动Gradio应用
    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=infer_ttswebui,
    )