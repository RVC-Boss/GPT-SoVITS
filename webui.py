import argparse
import json
import os
import platform
import re
import shutil
import traceback
from functools import partial
from multiprocessing import cpu_count
from subprocess import Popen
from typing import cast

import gradio as gr
import psutil
import torch
import yaml

from config import (
    GPU_INDEX,
    GPU_INFOS,
    IS_GPU,
    GPT_weight_root,
    GPT_weight_version2root,
    SoVITS_weight_root,
    SoVITS_weight_version2root,
    change_choices,
    exp_root,
    get_weights_names,
    infer_device,
    is_half,
    is_share,
    memset,
    pretrained_gpt_name,
    pretrained_sovits_name,
    python_exec,
    webui_port_infer_tts,
    webui_port_main,
    webui_port_subfix,
    webui_port_uvr5,
)
from GPT_SoVITS.Accelerate import (
    MLX,
    PyTorch,
    backends,
    console,
    logger,
    quantization_methods_mlx,
    quantization_methods_torch,
)
from tools import my_utils
from tools.asr.config import asr_dict
from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list
from tools.my_utils import check_details, check_for_existance

os.environ["version"] = version = "v2Pro"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTHONPATH"] = os.getcwd()


backends_gradio = [(b.replace("-", " "), b) for b in backends]

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
        return "en_US"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -s webui.py",
        description="python -s webui.py zh_CN",
    )
    p.add_argument(
        "language",
        nargs="?",
        default="Auto",
        type=lang_type,
        help="Language Code, Such as zh_CN, en-US",
    )
    return p


args = build_parser().parse_args()

tmp = "TEMP"
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if os.path.exists(tmp):
    for name in os.listdir(tmp):
        if name == "jieba.cache":
            continue
        path = f"{tmp}/{name}"
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            console.print(e)
            pass


language = str(args.language)
i18n = I18nAuto(language=language)
change_choice = partial(change_choices, i18n=i18n)

n_cpu = cpu_count()

set_gpu_numbers = GPU_INDEX
gpu_infos = GPU_INFOS
mem = memset
is_gpu_ok = IS_GPU

v3v4set = {"v3", "v4"}

sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"


def set_default():
    global \
        default_batch_size, \
        default_max_batch_size, \
        gpu_info, \
        default_sovits_epoch, \
        default_sovits_save_every_epoch, \
        max_sovits_epoch, \
        max_sovits_save_every_epoch, \
        default_batch_size_s1, \
        if_force_ckpt
    if_force_ckpt = False
    gpu_info = "\n".join(gpu_infos)
    if is_gpu_ok:
        minmem = min(mem)
        default_batch_size = minmem // 2 if version not in v3v4set else minmem // 8
        default_batch_size_s1 = minmem // 2
    else:
        default_batch_size = default_batch_size_s1 = int(psutil.virtual_memory().total / 1024 / 1024 / 1024 / 4)
    if version not in v3v4set:
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25  # 40
        max_sovits_save_every_epoch = 25  # 10
    else:
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 16  # 40 # 3 #训太多=作死
        max_sovits_save_every_epoch = 10  # 10 # 3

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3


set_default()

default_gpu_numbers = infer_device.index


def check_pretrained_is_exist(version):
    pretrained_model_list = (
        pretrained_sovits_name[version],
        pretrained_sovits_name[version].replace("s2G", "s2D"),
        pretrained_gpt_name[version],
        "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    )
    _ = ""
    for i in pretrained_model_list:
        if "s2Dv3" not in i and "s2Dv4" not in i and os.path.exists(i) is False:
            _ += f"\n    {i}"
    if _:
        logger.warning(i18n("以下模型不存在:") + _)


check_pretrained_is_exist(version)
for key in pretrained_sovits_name.keys():
    if os.path.exists(pretrained_sovits_name[key]) is False:
        pretrained_sovits_name[key] = ""
for key in pretrained_gpt_name.keys():
    if os.path.exists(pretrained_gpt_name[key]) is False:
        pretrained_gpt_name[key] = ""


for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)
SoVITS_names, GPT_names = get_weights_names(i18n)

p_label: Popen | None = None
p_uvr5: Popen | None = None
p_asr: Popen | None = None
p_denoise: Popen | None = None
p_tts_inference: Popen | None = None


def kill_process(pid: int, process_name=""):
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    for c in p.children(recursive=False):
        try:
            c.kill()
            c.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass

    try:
        p.kill()
        p.wait(timeout=5)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        pass

    console.print(process_name + i18n("进程已终止"))


def process_info(process_name="", indicator=""):
    if indicator == "opened":
        return process_name + i18n("已开启")
    elif indicator == "open":
        return i18n("开启") + process_name
    elif indicator == "closed":
        return process_name + i18n("已关闭")
    elif indicator == "close":
        return i18n("关闭") + process_name
    elif indicator == "running":
        return process_name + i18n("运行中")
    elif indicator == "occupy":
        return process_name + i18n("占用中") + "," + i18n("需先终止才能开启下一次任务")
    elif indicator == "finish":
        return process_name + i18n("已完成")
    elif indicator == "failed":
        return process_name + i18n("失败")
    elif indicator == "info":
        return process_name + i18n("进程输出信息")
    else:
        return process_name


process_name_subfix = i18n("音频标注WebUI")


def change_label(path_list):
    global p_label
    if p_label is None:
        check_for_existance([path_list])
        path_list = my_utils.clean_path(path_list)
        cmd = '"%s" -s tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s' % (
            python_exec,
            path_list,
            webui_port_subfix,
            is_share,
        )
        yield (
            process_info(process_name_subfix, "opened"),
            gr.update(visible=False),
            gr.update(visible=True),
        )
        console.print(cmd)
        p_label = Popen(cmd, shell=True)
    else:
        kill_process(p_label.pid, process_name_subfix)
        p_label = None
        yield (
            process_info(process_name_subfix, "closed"),
            gr.update(visible=True),
            gr.update(visible=False),
        )


process_name_uvr5 = i18n("人声分离WebUI")


def change_uvr5():
    global p_uvr5
    if p_uvr5 is None:
        cmd = '"%s" -s tools/uvr5/webui.py "%s" %s %s %s' % (
            python_exec,
            infer_device,
            is_half,
            webui_port_uvr5,
            is_share,
        )
        yield (
            process_info(process_name_uvr5, "opened"),
            gr.update(visible=False),
            gr.update(visible=True),
        )
        console.print(cmd)
        p_uvr5 = Popen(cmd, shell=True)
    else:
        kill_process(p_uvr5.pid, process_name_uvr5)
        p_uvr5 = None
        yield (
            process_info(process_name_uvr5, "closed"),
            gr.update(visible=True),
            gr.update(visible=False),
        )


process_name_tts = i18n("TTS推理WebUI")


def change_tts_inference(
    gpu_number: int,
    gpt_path: str,
    sovits_path: str,
    batched_infer_enabled: bool,
    backends_dropdown: str,
    quantization_methods_dropdown: str,
):
    global p_tts_inference
    env = os.environ.copy()
    cmd: list[str] = [python_exec, "-s", "-m"]
    if batched_infer_enabled:
        # fmt: off
        cmd.extend(
            [
                "GPT_SoVITS.inference_webui_fast", language,
                "-b", backends_dropdown,
                "-q", quantization_methods_dropdown,
                "-d", f"{infer_device.type}:{gpu_number}",
                "-p", str(webui_port_infer_tts),
                "--gpt", gpt_path,
                "--sovits", sovits_path,
            ]
        ) # fmt: on
    else:
        # fmt: off
        cmd.extend(
            [
                "GPT_SoVITS.inference_webui", language,
                "-b", backends_dropdown,
                "-q", quantization_methods_dropdown,
                "-d", f"{infer_device.type}:{gpu_number}",
                "-p", str(webui_port_infer_tts),
                "--gpt", gpt_path,
                "--sovits", sovits_path,
            ]
        ) # fmt: on

    if is_share:
        cmd.append("-s")

    yield (
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )

    if p_tts_inference is None:
        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.skip(),
        )
        console.print(" ".join(cmd))
        p_tts_inference = Popen(cmd, env=env)
    else:
        kill_process(p_tts_inference.pid, process_name_tts)
        p_tts_inference = None
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.skip(),
        )


process_name_asr = i18n("语音识别")


def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision):
    global p_asr
    if p_asr is None:
        asr_inp_dir = my_utils.clean_path(asr_inp_dir)
        asr_opt_dir = my_utils.clean_path(asr_opt_dir)
        check_for_existance([asr_inp_dir])
        cmd = f'"{python_exec}" -s tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f" -s {asr_model_size}"
        cmd += f" -l {asr_lang}"
        cmd += f" -p {asr_precision}"
        output_file_name = os.path.basename(asr_inp_dir)
        output_folder = asr_opt_dir or "output/asr_opt"
        output_file_path = os.path.abspath(f"{output_folder}/{output_file_name}.list")
        yield (
            process_info(process_name_asr, "opened"),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        console.print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None
        yield (
            process_info(process_name_asr, "finish"),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=output_file_path),
            gr.update(value=output_file_path),
            gr.update(value=asr_inp_dir),
        )
    else:
        yield (
            process_info(process_name_asr, "occupy"),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )


def close_asr():
    global p_asr
    if p_asr is not None:
        kill_process(p_asr.pid, process_name_asr)
        p_asr = None
    return (
        process_info(process_name_asr, "closed"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


p_train_SoVITS: Popen | None = None
process_name_sovits = i18n("SoVITS训练")


def open1Ba(
    version,
    batch_size,
    total_epoch,
    exp_name,
    text_low_lr_rate,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers1Ba,
    pretrained_s2G,
    pretrained_s2D,
    if_grad_ckpt,
    lora_rank,
):
    global p_train_SoVITS
    if p_train_SoVITS is None:
        exp_name = exp_name.rstrip(" ")
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        with open(config_file) as f:
            config = f.read()
            data: dict = json.loads(config)
        s2_dir = f"{exp_root}/{exp_name}"
        os.makedirs(f"{s2_dir}/logs_s2_{version}", exist_ok=True)
        if check_for_existance([s2_dir], is_train=True):
            check_details([s2_dir], is_train=True)
        if is_half is False:
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["grad_ckpt"] = if_grad_ckpt
        data["train"]["lora_rank"] = lora_rank
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_version2root[version]
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = f"{tmp}/tmp_s2.json"
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_numbers1Ba).strip("[]").replace(" ", "")

        if version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
            cmd = [
                python_exec,
                "-s",
                "GPT_SoVITS/s2_train.py",
                "--config",
                tmp_config_path,
            ]
        else:
            cmd = [
                python_exec,
                "-s",
                "GPT_SoVITS/s2_train_v3_lora.py",
                "--config",
                tmp_config_path,
            ]
        console.print(" ".join(cmd))

        p = Popen(cmd, env=env)
        p_train_SoVITS = p

        yield (
            process_info(process_name_sovits, "opened"),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.skip(),
            gr.skip(),
        )

        code = p.wait()
        p_train_SoVITS = None

        if code == 0:
            yield (
                process_info(process_name_sovits, "finish"),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.skip(),
                gr.skip(),
            )
        else:
            yield (
                process_info(process_name_sovits, "failed"),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.skip(),
                gr.skip(),
            )
            return (gr.skip() for i in range(5))

        SoVITS_dropdown_update, GPT_dropdown_update = change_choice()

        yield (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            SoVITS_dropdown_update,
            GPT_dropdown_update,
        )
    else:
        yield (
            process_info(process_name_sovits, "occupy"),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.skip(),
            gr.skip(),
        )


def close1Ba():
    global p_train_SoVITS
    if p_train_SoVITS:
        kill_process(p_train_SoVITS.pid, process_name_sovits)
        p_train_SoVITS = None
    return (
        process_info(process_name_sovits, "closed"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


p_train_GPT: Popen | None = None
process_name_gpt = i18n("GPT训练")


def open1Bb(
    batch_size,
    total_epoch,
    exp_name,
    if_dpo,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers,
    pretrained_s1,
):
    global p_train_GPT
    if p_train_GPT is None:
        exp_name = exp_name.rstrip(" ")
        with open(
            "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml"
        ) as f:
            config = f.read()
            data: dict = yaml.load(config, Loader=yaml.FullLoader)
        s1_dir = f"{exp_root}/{exp_name}"
        os.makedirs(f"{s1_dir}/logs_s1", exist_ok=True)
        if check_for_existance([s1_dir], is_train=True):
            check_details([s1_dir], is_train=True)

        if is_half is False or torch.mps.is_available():
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_version2root[version]
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = f"{s1_dir}/6-name2semantic.tsv"
        data["train_phoneme_path"] = f"{s1_dir}/2-name2text.txt"
        data["output_dir"] = f"{s1_dir}/logs_s1_{version}"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_numbers).strip("[]").replace(" ", "")

        tmp_config_path = f"{tmp}/tmp_s1.yaml"
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))

        cmd = [python_exec, "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp_config_path]

        console.print(" ".join(cmd))

        p = Popen(cmd, env=env)
        p_train_GPT = p

        yield (
            process_info(process_name_gpt, "opened"),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.skip(),
            gr.skip(),
        )

        code = p.wait()
        p_train_GPT = None

        if code == 0:
            yield (
                process_info(process_name_gpt, "finish"),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.skip(),
                gr.skip(),
            )
        else:
            yield (
                process_info(process_name_gpt, "failed"),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.skip(),
                gr.skip(),
            )
            return (gr.skip() for i in range(5))

        SoVITS_dropdown_update, GPT_dropdown_update = change_choice()

        yield (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            SoVITS_dropdown_update,
            GPT_dropdown_update,
        )
    else:
        yield (
            process_info(process_name_gpt, "occupy"),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.skip(),
            gr.skip(),
        )


def close1Bb():
    global p_train_GPT
    if p_train_GPT is not None:
        kill_process(p_train_GPT.pid, process_name_gpt)
        p_train_GPT = None
    return (
        process_info(process_name_gpt, "closed"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


ps_slice = []
process_name_slice = i18n("语音切分")


def open_slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_existance([inp])
    if os.path.exists(inp) is False:
        yield (
            i18n("输入路径不存在"),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.skip(),
            gr.skip(),
        )
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        yield (
            i18n("输入路径存在但不可用"),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.skip(),
            gr.skip(),
        )
        return
    if ps_slice == []:
        for i_part in range(n_parts):
            cmd = '"%s" -s tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s' % (
                python_exec,
                inp,
                opt_root,
                threshold,
                min_length,
                min_interval,
                hop_size,
                max_sil_kept,
                _max,
                alpha,
                i_part,
                n_parts,
            )
            console.print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield (
            process_info(process_name_slice, "opened"),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.skip(),
            gr.skip(),
        )
        for p in ps_slice:
            p.wait()
        ps_slice = []
        yield (
            process_info(process_name_slice, "finish"),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=opt_root),
            gr.update(value=opt_root),
        )
    else:
        yield (
            process_info(process_name_slice, "occupy"),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.skip(),
            gr.skip(),
        )


def close_slice():
    global ps_slice
    if ps_slice != []:
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid, process_name_slice)
            except Exception as _:
                traceback.print_exc()
        ps_slice = []
    return (
        process_info(process_name_slice, "closed"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


ps1a: None | Popen = None
process_name_1a = i18n("文本分词与特征提取")


def open1a(
    inp_text: str,
    inp_wav_dir: str,
    exp_name: str,
    gpu_numbers: list[int],
    bert_pretrained_dir: str,
    version: str,
    nproc: int = 1,
):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1a is None:
        opt_dir = f"{exp_root}/{exp_name}"

        env = os.environ.copy()

        # fmt: off
        cmd = [
            python_exec, "-s", "-m", "GPT_SoVITS.prepare_datasets.1_get_text",
            "--inp-list", inp_text,
            "--opt", opt_dir,
            "--bert", bert_pretrained_dir,
            "--version", version,
            "--device", infer_device.type,
            "--device-id", str(gpu_numbers).strip("[]").replace(" ",""),
            "--nproc", str(nproc),
        ] 
        # fmt: on

        if is_half:
            cmd.append("--fp16")
        else:
            cmd.append("--no-fp16")

        console.print(" ".join(cmd))
        p = Popen(cmd, env=env)

        yield (
            process_info(process_name_1a, "running"),
            gr.update(visible=False),
            gr.update(visible=True),
        )

        code = p.wait()
        ps1a = None

        if code == 0:
            yield (
                process_info(process_name_1a, "finish"),
                gr.update(visible=True),
                gr.update(visible=False),
            )
        else:
            yield (
                process_info(process_name_1a, "failed"),
                gr.update(visible=True),
                gr.update(visible=False),
            )
    else:
        yield (
            process_info(process_name_1a, "occupy"),
            gr.update(visible=False),
            gr.update(visible=True),
        )


def close1a():
    global ps1a
    if ps1a:
        try:
            kill_process(ps1a.pid, process_name_1a)
        except Exception as _:
            traceback.print_exc()
        ps1a = None
    return (
        process_info(process_name_1a, "closed"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


ps1b: None | Popen = None
process_name_1b = i18n("语音自监督特征提取")


def open1b(
    version: str,
    inp_text: str,
    inp_wav_dir: str,
    exp_name: str,
    gpu_numbers: list[int],
    ssl_pretrained_dir: str,
    nproc: int = 1,
):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1b is None:
        opt_dir = f"{exp_root}/{exp_name}"

        env = os.environ.copy()

        # fmt: off
        cmd = [
            python_exec, "-s", "-m", "GPT_SoVITS.prepare_datasets.2_get_hubert_sv_wav32k",
            "--inp-list", inp_text,
            "--opt", opt_dir,
            "--cnhubert", ssl_pretrained_dir,
            "--device", infer_device.type,
            "--device-id", str(gpu_numbers).strip("[]").replace(" ",""),
            "--nproc", str(nproc),
        ] 
        # fmt: on

        if inp_wav_dir:
            cmd.extend(["--wav-dir", inp_wav_dir])

        if "Pro" in version:
            cmd.extend(["--sv", sv_path])

        if is_half:
            cmd.append("--fp16")
        else:
            cmd.append("--no-fp16")

        console.print(" ".join(cmd))
        p = Popen(cmd, env=env)

        ps1b = p

        yield (
            process_info(process_name_1b, "running"),
            gr.update(visible=False),
            gr.update(visible=True),
        )

        code = p.wait()
        ps1b = None

        if code == 0:
            yield (
                process_info(process_name_1b, "finish"),
                gr.update(visible=True),
                gr.update(visible=False),
            )
        else:
            yield (
                process_info(process_name_1b, "failed"),
                gr.update(visible=True),
                gr.update(visible=False),
            )
    else:
        yield (
            process_info(process_name_1b, "occupy"),
            gr.update(visible=False),
            gr.update(visible=True),
        )


def close1b():
    global ps1b
    if ps1b:
        try:
            kill_process(ps1b.pid, process_name_1b)
        except Exception as _:
            traceback.print_exc()
        ps1b = None
    return (
        process_info(process_name_1b, "closed"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


ps1c: None | Popen = None
process_name_1c = i18n("语义Token提取")


def open1c(
    inp_text: str,
    exp_name: str,
    gpu_numbers: list[int],
    pretrained_s2G_path: str,
    nproc: int = 1,
):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    check_for_existance([inp_text], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1c is None:
        opt_dir = f"{exp_root}/{exp_name}"

        env = os.environ.copy()

        # fmt: off
        cmd = [
            python_exec, "-s", "-m", "GPT_SoVITS.prepare_datasets.3_get_semantic",
            "--inp-list", inp_text,
            "--opt", opt_dir,
            "--pretrained-s2g", pretrained_s2G_path,
            "--device", infer_device.type,
            "--device-id", str(gpu_numbers).strip("[]").replace(" ",""),
            "--nproc", str(nproc),
        ] 
        # fmt: on

        if is_half:
            cmd.append("--fp16")
        else:
            cmd.append("--no-fp16")

        console.print(" ".join(cmd))
        p = Popen(cmd, env=env)

        ps1c = p

        yield (
            process_info(process_name_1c, "running"),
            gr.update(visible=False),
            gr.update(visible=True),
        )

        code = p.wait()
        ps1c = None

        if code == 0:
            yield (
                process_info(process_name_1c, "finish"),
                gr.update(visible=True),
                gr.update(visible=False),
            )
        else:
            yield (
                process_info(process_name_1c, "failed"),
                gr.update(visible=True),
                gr.update(visible=False),
            )

    else:
        yield (
            process_info(process_name_1c, "occupy"),
            gr.update(visible=False),
            gr.update(visible=True),
        )


def close1c():
    global ps1c
    if ps1c:
        try:
            kill_process(ps1c.pid, process_name_1c)
        except Exception as _:
            traceback.print_exc()
    ps1c = None
    return (
        process_info(process_name_1c, "closed"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


ps1abc: list[None | Popen] = [None] * 3
process_name_1abc = i18n("训练集格式化一键三连")


def open1abc(
    version: str,
    inp_text: str,
    inp_wav_dir: str,
    exp_name: str,
    gpu_numbers_1: list[int],
    gpu_numbers_2: list[int],
    gpu_numbers_3: list[int],
    bert_pretrained_dir: str,
    ssl_pretrained_dir: str,
    pretrained_s2G_path: str,
    nproc: int = 1,
):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if not all(ps1abc):
        opt_dir = f"{exp_root}/{exp_name}"

        env = os.environ.copy()

        # Step 1
        # fmt: off
        cmd_1 = [
            python_exec, "-s", "-m", "GPT_SoVITS.prepare_datasets.1_get_text",
            "--inp-list", inp_text,
            "--opt", opt_dir,
            "--bert", bert_pretrained_dir,
            "--version", version,
            "--device", infer_device.type,
            "--device-id", str(gpu_numbers_1).strip("[]").replace(" ",""),
            "--nproc", str(nproc),
        ] 
        # fmt: on

        if is_half:
            cmd_1.append("--fp16")
        else:
            cmd_1.append("--no-fp16")

        console.print(" ".join(cmd_1))
        p = Popen(cmd_1, env=env)
        ps1abc[0] = p

        yield (
            i18n("进度") + ": 1A-Doing",
            gr.update(visible=False),
            gr.update(visible=True),
        )

        code = p.wait()
        ps1abc[0] = None

        if code == 0:
            yield (
                i18n("进度") + ": 1A-Done",
                gr.update(visible=False),
                gr.update(visible=True),
            )
        else:
            yield (
                i18n("进度") + ": 1A-Failed",
                gr.update(visible=True),
                gr.update(visible=False),
            )
            return (gr.skip() for i in range(3))

        # Step 2
        # fmt: off
        cmd_2 = [
            python_exec, "-s", "-m", "GPT_SoVITS.prepare_datasets.2_get_hubert_sv_wav32k",
            "--inp-list", inp_text,
            "--opt", opt_dir,
            "--cnhubert", ssl_pretrained_dir,
            "--device", infer_device.type,
            "--device-id", str(gpu_numbers_2).strip("[]").replace(" ",""),
            "--nproc", str(nproc),
        ]  
        # fmt: on

        if inp_wav_dir:
            cmd_2.extend(["--wav-dir", inp_wav_dir])

        if "Pro" in version:
            cmd_2.extend(["--sv", sv_path])

        if is_half:
            cmd_2.append("--fp16")
        else:
            cmd_2.append("--no-fp16")

        console.print(" ".join(cmd_2))
        p = Popen(cmd_2, env=env)
        ps1abc[1] = p

        yield (
            i18n("进度") + ": 1A-Done, 1B-Doing",
            gr.update(visible=False),
            gr.update(visible=True),
        )

        code = p.wait()
        ps1abc[1] = None

        if code == 0:
            yield (
                i18n("进度") + ": 1A-Done, 1B-Done",
                gr.update(visible=False),
                gr.update(visible=True),
            )
        else:
            yield (
                i18n("进度") + ": 1A-Done, 1B-Failed",
                gr.update(visible=True),
                gr.update(visible=False),
            )
            return (gr.skip() for i in range(3))

        # Step 3
        # fmt: off
        cmd_3 = [
            python_exec, "-s", "-m", "GPT_SoVITS.prepare_datasets.3_get_semantic",
            "--inp-list", inp_text,
            "--opt", opt_dir,
            "--pretrained-s2g", pretrained_s2G_path,
            "--device", infer_device.type,
            "--device-id", str(gpu_numbers_3).strip("[]").replace(" ",""),
            "--nproc", str(nproc),
        ] 
        # fmt: on

        if is_half:
            cmd_3.append("--fp16")
        else:
            cmd_3.append("--no-fp16")

        console.print(" ".join(cmd_3))
        p = Popen(cmd_3, env=env)
        ps1abc[2] = p

        yield (
            i18n("进度") + ": 1A-Done, 1B-Done, 1C-Doing",
            gr.update(visible=False),
            gr.update(visible=True),
        )

        code = p.wait()
        ps1abc[2] = None

        if code == 0:
            yield (
                process_info(process_name_1abc, "finish"),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        else:
            yield (
                i18n("进度") + ": 1A-Done, 1B-Done, 1C-Failed",
                gr.update(visible=True),
                gr.update(visible=False),
            )
            return (gr.skip() for i in range(3))

    else:
        yield (
            process_info(process_name_1abc, "occupy"),
            gr.update(visible=False),
            gr.update(visible=True),
        )


def close1abc():
    global ps1abc
    if any(ps1abc):
        for p1abc in ps1abc:
            if p1abc is None:
                continue
            try:
                kill_process(p1abc.pid, process_name_1abc)
            except Exception as _:
                traceback.print_exc()
        ps1abc = [None] * 3
    return (
        process_info(process_name_1abc, "closed"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def switch_version(version_):
    os.environ["version"] = version_
    global version
    version = version_
    if pretrained_sovits_name[version] != "" and pretrained_gpt_name[version] != "":
        ...
    else:
        gr.Warning(i18n("未下载模型") + ": " + version.upper())
    set_default()
    return (
        gr.update(value=pretrained_sovits_name[version]),
        gr.update(value=pretrained_sovits_name[version].replace("s2G", "s2D")),
        gr.update(value=pretrained_gpt_name[version]),
        gr.update(value=pretrained_gpt_name[version]),
        gr.update(value=pretrained_sovits_name[version]),
        gr.update(value=default_batch_size, maximum=default_max_batch_size),
        gr.update(value=default_sovits_epoch, maximum=max_sovits_epoch),
        gr.update(value=default_sovits_save_every_epoch, maximum=max_sovits_save_every_epoch),
        gr.update(visible=False if version in v3v4set else True),
        gr.update(
            visible=False if version not in v3v4set else True,
            value=False if not if_force_ckpt else True,
            interactive=True if not if_force_ckpt else False,
        ),
        gr.update(value=False, interactive=True),
        gr.update(visible=True if version in v3v4set else False),
    )


def sync(text):
    return gr.update(value=text)


def changeQuantization(backend: str, gradio_call=True):
    backend = backend.lower().replace("-", "_")
    if backend in MLX.backends:
        choices = quantization_methods_mlx
    elif backend in PyTorch.backends:
        choices = quantization_methods_torch
    else:
        choices = ["None"]

    choices = [str(c) for c in choices]

    if gradio_call:
        return gr.update(choices=choices, value="None")
    else:
        return choices


GPU_INDEX.add(0)
GPU_INDEX_LIST = list(GPU_INDEX)
GPU_INDEX_LIST.sort()
with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:
    gr.HTML(
        top_html.format(
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.")
            + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        ),
        elem_classes="markdown",
    )

    with gr.Tabs():
        with gr.TabItem("0-" + i18n("前置数据集获取工具")):
            with gr.Accordion(label="0a-" + i18n("UVR5人声伴奏分离&去混响去延迟工具")):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        with gr.Row(equal_height=True):
                            uvr5_info = gr.Textbox(label=process_info(process_name_uvr5, "info"))
                    open_uvr5 = gr.Button(
                        value=process_info(process_name_uvr5, "open"), variant="primary", visible=True
                    )
                    close_uvr5 = gr.Button(
                        value=process_info(process_name_uvr5, "close"), variant="primary", visible=False
                    )

            with gr.Accordion(label="0b-" + i18n("语音切分工具")):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        with gr.Row(equal_height=True):
                            slice_inp_path = gr.Textbox(
                                label=i18n("音频自动切分输入路径, 可文件可文件夹"),
                                placeholder="D:/InputAudioFolder"
                                if platform.system() == "Windows"
                                else "~/InputAudioFolder",
                            )
                            slice_opt_root = gr.Textbox(
                                label=i18n("切分后的子音频的输出根目录"), value="output/slicer_opt"
                            )
                        with gr.Row(equal_height=True):
                            threshold = gr.Textbox(
                                label=i18n("threshold:音量小于这个值视作静音的备选切割点"), value="-34"
                            )
                            min_length = gr.Textbox(
                                label=i18n("min_length: 每段最小多长, 如果第一段太短一直和后面段连起来直到超过这个值"),
                                value="4000",
                            )
                            min_interval = gr.Textbox(label=i18n("min_interval:最短切割间隔"), value="300")
                            hop_size = gr.Textbox(
                                label=i18n("hop_size: 怎么算音量曲线, 越小精度越大计算量越高 (不是精度越大效果越好)"),
                                value="10",
                            )
                            max_sil_kept = gr.Textbox(label=i18n("max_sil_kept:切完后静音最多留多长"), value="500")
                        with gr.Row(equal_height=True):
                            _max = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label=i18n("max:归一化后最大值多少"),
                                value=0.9,
                                interactive=True,
                            )
                            alpha = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label=i18n("alpha_mix:混多少比例归一化后音频进来"),
                                value=0.25,
                                interactive=True,
                            )
                        with gr.Row(equal_height=True):
                            n_process = gr.Slider(
                                minimum=1,
                                maximum=n_cpu,
                                step=1,
                                label=i18n("切割使用的进程数"),
                                value=4,
                                interactive=True,
                            )
                            slicer_info = gr.Textbox(label=process_info(process_name_slice, "info"))
                    open_slicer_button = gr.Button(
                        value=process_info(process_name_slice, "open"), variant="primary", visible=True
                    )
                    close_slicer_button = gr.Button(
                        value=process_info(process_name_slice, "close"), variant="primary", visible=False
                    )

            with gr.Accordion(label="0c-" + i18n("语音识别工具")):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        with gr.Row(equal_height=True):
                            asr_inp_dir = gr.Textbox(
                                label=i18n("输入文件夹路径"),
                                value="output/silcer_opt",
                                interactive=True,
                            )
                            asr_opt_dir = gr.Textbox(
                                label=i18n("输出文件夹路径"), value="output/asr_opt", interactive=True
                            )
                        with gr.Row(equal_height=True):
                            asr_model = gr.Dropdown(
                                label=i18n("ASR 模型"),
                                choices=list(asr_dict.keys()),
                                interactive=True,
                                value="达摩 ASR (中文)",
                            )
                            asr_size = gr.Dropdown(
                                label=i18n("ASR 模型尺寸"), choices=["large"], interactive=True, value="large"
                            )
                            asr_lang = gr.Dropdown(
                                label=i18n("ASR 语言设置"), choices=["zh", "yue"], interactive=True, value="zh"
                            )
                            asr_precision = gr.Dropdown(
                                label=i18n("数据类型精度"), choices=["float32"], interactive=True, value="float32"
                            )
                        with gr.Row(equal_height=True):
                            asr_info = gr.Textbox(label=process_info(process_name_asr, "info"))
                    open_asr_button = gr.Button(
                        value=process_info(process_name_asr, "open"), variant="primary", visible=True
                    )
                    close_asr_button = gr.Button(
                        value=process_info(process_name_asr, "close"), variant="primary", visible=False
                    )

                def change_lang_choices(key):  # 根据选择的模型修改可选的语言
                    return gr.update(value=asr_dict[key]["lang"][0], choices=asr_dict[key]["lang"])

                def change_size_choices(key):  # 根据选择的模型修改可选的模型尺寸
                    return gr.update(value=asr_dict[key]["size"][-1], choices=asr_dict[key]["size"])

                def change_precision_choices(key):  # 根据选择的模型修改可选的语言
                    if key == "Faster Whisper (多语种)":
                        if default_batch_size <= 4:
                            precision = "int8"
                        elif is_half:
                            precision = "float16"
                        else:
                            precision = "float32"
                    else:
                        precision = "float32"
                    return gr.update(value=precision, choices=asr_dict[key]["precision"])

                asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                asr_model.change(change_size_choices, [asr_model], [asr_size])
                asr_model.change(change_precision_choices, [asr_model], [asr_precision])

            with gr.Accordion(label="0d-" + i18n("语音文本校对标注工具")):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        with gr.Row(equal_height=True):
                            path_list = gr.Textbox(
                                label=i18n("标注文件路径 (含文件后缀 *.list)"),
                                value="output/asr_opt/slicer_opt.list",
                                interactive=True,
                            )
                            label_info = gr.Textbox(label=process_info(process_name_subfix, "info"))
                    open_label = gr.Button(
                        value=process_info(process_name_subfix, "open"), variant="primary", visible=True
                    )
                    close_label = gr.Button(
                        value=process_info(process_name_subfix, "close"), variant="primary", visible=False
                    )

                open_label.click(change_label, [path_list], [label_info, open_label, close_label])
                close_label.click(change_label, [path_list], [label_info, open_label, close_label])
                open_uvr5.click(change_uvr5, [], [uvr5_info, open_uvr5, close_uvr5])
                close_uvr5.click(change_uvr5, [], [uvr5_info, open_uvr5, close_uvr5])

        with gr.TabItem(i18n("1-GPT-SoVITS-TTS")):
            with gr.Accordion(i18n("微调模型信息")):
                with gr.Row(equal_height=True):
                    with gr.Column():
                        exp_name = gr.Textbox(
                            label=i18n("实验/模型名"),
                            value="xxx",
                            interactive=True,
                            scale=3,
                        )
                    with gr.Column():
                        gpu_info_box = gr.Textbox(
                            label=i18n("显卡信息"),
                            value=gpu_info,
                            visible=True,
                            interactive=False,
                            scale=5,
                        )
                    with gr.Column():
                        version_checkbox = gr.Dropdown(
                            label=i18n("训练模型的版本"),
                            value=version,
                            choices=[
                                ("V1", "v1"),
                                ("V2", "v2"),
                                ("V4", "v4"),
                                ("V2 Pro", "v2Pro"),
                                ("V2 Pro Plus", "v2ProPlus"),
                            ],
                            scale=5,
                        )
                    with gr.Column():
                        n_processes = gr.Slider(0, 6, 2, step=1, label=i18n("每卡预处理进程数"))

            with gr.Accordion(label=i18n("预训练模型路径"), open=False):
                with gr.Row(equal_height=True):
                    with gr.Row(equal_height=True):
                        pretrained_s1 = gr.Textbox(
                            label=i18n("预训练GPT模型路径"),
                            value=pretrained_gpt_name[version],
                            interactive=True,
                            lines=1,
                            max_lines=1,
                            scale=3,
                        )
                        pretrained_s2G = gr.Textbox(
                            label=i18n("预训练SoVITS-G模型路径"),
                            value=pretrained_sovits_name[version],
                            interactive=True,
                            lines=1,
                            max_lines=1,
                            scale=5,
                        )
                        pretrained_s2D = gr.Textbox(
                            label=i18n("预训练SoVITS-D模型路径"),
                            value=pretrained_sovits_name[version].replace("s2G", "s2D"),
                            interactive=True,
                            lines=1,
                            max_lines=1,
                            scale=5,
                        )

            with gr.TabItem("1A-" + i18n("训练集格式化工具")):
                with gr.Accordion(label=i18n("输出logs/实验名目录下应有23456开头的文件和文件夹")):
                    with gr.Row(equal_height=True):
                        with gr.Row(equal_height=True):
                            inp_text = gr.Textbox(
                                label=i18n("*文本标注文件"),
                                value=r"output/asr_opt/slicer_opt.list",
                                interactive=True,
                                scale=10,
                            )
                        with gr.Row(equal_height=True):
                            inp_wav_dir = gr.Textbox(
                                label=i18n("*训练集音频文件目录"),
                                # value=r"D:\RVC1006\GPT-SoVITS\raw\xxx",
                                interactive=True,
                                placeholder=i18n(
                                    "填切割后音频所在目录! 读取的音频文件完整路径=该目录-拼接-list文件里波形对应的文件名 (不是全路径). 如果留空则使用.list文件里的绝对全路径."
                                ),
                                scale=10,
                            )

                with gr.Accordion(label="1Aa-" + process_name_1a):
                    with gr.Row(equal_height=True):
                        with gr.Row(equal_height=True):
                            gpu_numbers1a = gr.Dropdown(
                                label=i18n("GPU卡号"),
                                choices=GPU_INDEX_LIST,
                                value=GPU_INDEX_LIST,
                                interactive=True,
                                multiselect=True,
                                allow_custom_value=False,
                            )
                        with gr.Row(equal_height=True):
                            bert_pretrained_dir = gr.Textbox(
                                label=i18n("预训练中文BERT模型路径"),
                                value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                                interactive=False,
                                lines=2,
                            )
                        with gr.Row(equal_height=True):
                            button1a_open = gr.Button(
                                value=process_info(process_name_1a, "open"), variant="primary", visible=True
                            )
                            button1a_close = gr.Button(
                                value=process_info(process_name_1a, "close"), variant="primary", visible=False
                            )
                        with gr.Row(equal_height=True):
                            info1a = gr.Textbox(label=process_info(process_name_1a, "info"))

                with gr.Accordion(label="1Ab-" + process_name_1b):
                    with gr.Row(equal_height=True):
                        with gr.Row(equal_height=True):
                            gpu_numbers1b = gr.Dropdown(
                                label=i18n("GPU卡号"),
                                choices=GPU_INDEX_LIST,
                                value=GPU_INDEX_LIST,
                                interactive=True,
                                multiselect=True,
                                allow_custom_value=False,
                            )
                        with gr.Row(equal_height=True):
                            cnhubert_base_dir = gr.Textbox(
                                label=i18n("预训练SSL模型路径"),
                                value="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                                interactive=False,
                                lines=2,
                            )
                        with gr.Row(equal_height=True):
                            button1b_open = gr.Button(
                                value=process_info(process_name_1b, "open"), variant="primary", visible=True
                            )
                            button1b_close = gr.Button(
                                value=process_info(process_name_1b, "close"), variant="primary", visible=False
                            )
                        with gr.Row(equal_height=True):
                            info1b = gr.Textbox(label=process_info(process_name_1b, "info"))

                with gr.Accordion(label="1Ac-" + process_name_1c):
                    with gr.Row(equal_height=True):
                        with gr.Row(equal_height=True):
                            gpu_numbers1c = gr.Dropdown(
                                label=i18n("GPU卡号"),
                                choices=GPU_INDEX_LIST,
                                value=GPU_INDEX_LIST,
                                interactive=True,
                                multiselect=True,
                                allow_custom_value=False,
                            )
                        with gr.Row(equal_height=True):
                            pretrained_s2G_ = gr.Textbox(
                                label=i18n("预训练SoVITS-G模型路径"),
                                value=pretrained_sovits_name[version],
                                interactive=False,
                                lines=2,
                            )
                        with gr.Row(equal_height=True):
                            button1c_open = gr.Button(
                                value=process_info(process_name_1c, "open"), variant="primary", visible=True
                            )
                            button1c_close = gr.Button(
                                value=process_info(process_name_1c, "close"), variant="primary", visible=False
                            )
                        with gr.Row(equal_height=True):
                            info1c = gr.Textbox(label=process_info(process_name_1c, "info"))

                with gr.Accordion(label="1Aabc-" + process_name_1abc):
                    with gr.Row(equal_height=True):
                        with gr.Row(equal_height=True):
                            button1abc_open = gr.Button(
                                value=process_info(process_name_1abc, "open"), variant="primary", visible=True
                            )
                            button1abc_close = gr.Button(
                                value=process_info(process_name_1abc, "close"), variant="primary", visible=False
                            )
                        with gr.Row(equal_height=True):
                            info1abc = gr.Textbox(label=process_info(process_name_1abc, "info"))

            pretrained_s2G.change(sync, [pretrained_s2G], [pretrained_s2G_])
            open_asr_button.click(
                open_asr,
                [asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang, asr_precision],
                [asr_info, open_asr_button, close_asr_button, path_list, inp_text, inp_wav_dir],
            )
            close_asr_button.click(close_asr, [], [asr_info, open_asr_button, close_asr_button])
            open_slicer_button.click(
                open_slice,
                [
                    slice_inp_path,
                    slice_opt_root,
                    threshold,
                    min_length,
                    min_interval,
                    hop_size,
                    max_sil_kept,
                    _max,
                    alpha,
                    n_process,
                ],
                [slicer_info, open_slicer_button, close_slicer_button, asr_inp_dir, inp_wav_dir],
            )
            close_slicer_button.click(close_slice, [], [slicer_info, open_slicer_button, close_slicer_button])

            button1a_open.click(
                open1a,
                [inp_text, inp_wav_dir, exp_name, gpu_numbers1a, bert_pretrained_dir, version_checkbox, n_processes],
                [info1a, button1a_open, button1a_close],
            )
            button1a_close.click(close1a, [], [info1a, button1a_open, button1a_close])
            button1b_open.click(
                open1b,
                [version_checkbox, inp_text, inp_wav_dir, exp_name, gpu_numbers1b, cnhubert_base_dir, n_processes],
                [info1b, button1b_open, button1b_close],
            )
            button1b_close.click(close1b, [], [info1b, button1b_open, button1b_close])
            button1c_open.click(
                open1c,
                [inp_text, exp_name, gpu_numbers1c, pretrained_s2G, n_processes],
                [info1c, button1c_open, button1c_close],
            )
            button1c_close.click(close1c, [], [info1c, button1c_open, button1c_close])
            button1abc_open.click(
                open1abc,
                [
                    version_checkbox,
                    inp_text,
                    inp_wav_dir,
                    exp_name,
                    gpu_numbers1a,
                    gpu_numbers1b,
                    gpu_numbers1c,
                    bert_pretrained_dir,
                    cnhubert_base_dir,
                    pretrained_s2G,
                    n_processes,
                ],
                [info1abc, button1abc_open, button1abc_close],
            )
            button1abc_close.click(close1abc, [], [info1abc, button1abc_open, button1abc_close])

            with gr.TabItem("1B-" + i18n("微调训练")):
                with gr.Accordion(label="1Ba-" + i18n("SoVITS 训练: 模型权重文件在 SoVITS_weights/")):
                    with gr.Row(equal_height=True):
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=default_max_batch_size,
                            step=1,
                            label=i18n("每张显卡的batch_size"),
                            value=default_batch_size,
                            interactive=True,
                        )
                        total_epoch = gr.Slider(
                            minimum=1,
                            maximum=max_sovits_epoch,
                            step=1,
                            label=i18n("总训练轮数total_epoch, 不建议太高"),
                            value=default_sovits_epoch,
                            interactive=True,
                        )
                        with gr.Column(scale=2):
                            if_save_latest = gr.Checkbox(
                                label=i18n("是否仅保存最新的权重文件以节省硬盘空间"),
                                value=True,
                                interactive=True,
                                show_label=True,
                            )
                            if_save_every_weights = gr.Checkbox(
                                label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                                value=True,
                                interactive=True,
                                show_label=True,
                            )
                            if_grad_ckpt = gr.Checkbox(
                                label="v3是否开启梯度检查点节省显存占用",
                                value=False,
                                interactive=True if version in v3v4set else False,
                                show_label=True,
                                visible=False,
                            )  # 只有V3s2可以用
                    with gr.Row(equal_height=True):
                        text_low_lr_rate = gr.Slider(
                            minimum=0.2,
                            maximum=0.6,
                            step=0.05,
                            label=i18n("文本模块学习率权重"),
                            value=0.4,
                            visible=True if version not in v3v4set else False,
                        )  # v3v4 not need
                        lora_rank = gr.Radio(
                            label=i18n("LoRA秩"),
                            value="32",
                            choices=["16", "32", "64", "128"],
                            visible=True if version in v3v4set else False,
                        )  # v1v2 not need
                        save_every_epoch = gr.Slider(
                            minimum=1,
                            maximum=max_sovits_save_every_epoch,
                            step=1,
                            label=i18n("保存频率save_every_epoch"),
                            value=default_sovits_save_every_epoch,
                            interactive=True,
                        )
                        with gr.Column(scale=3):
                            gpu_numbers1Ba = gr.Dropdown(
                                label=i18n("GPU卡号"),
                                choices=GPU_INDEX_LIST,
                                value=GPU_INDEX_LIST,
                                interactive=True,
                                multiselect=True,
                                allow_custom_value=False,
                            )
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            button1Ba_open = gr.Button(
                                value=process_info(process_name_sovits, "open"), variant="primary", visible=True
                            )
                            button1Ba_close = gr.Button(
                                value=process_info(process_name_sovits, "close"), variant="primary", visible=False
                            )
                        with gr.Column():
                            info1Ba = gr.Textbox(label=process_info(process_name_sovits, "info"))
                with gr.Accordion(label="1Bb-" + i18n("GPT 训练: 模型权重文件在 GPT_weights/")):
                    with gr.Row(equal_height=True):
                        batch_size1Bb = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label=i18n("每张显卡的batch_size"),
                            value=default_batch_size_s1,
                            interactive=True,
                        )
                        total_epoch1Bb = gr.Slider(
                            minimum=2,
                            maximum=50,
                            step=1,
                            label=i18n("总训练轮数total_epoch"),
                            value=15,
                            interactive=True,
                        )
                        with gr.Column(scale=2):
                            if_save_latest1Bb = gr.Checkbox(
                                label=i18n("是否仅保存最新的权重文件以节省硬盘空间"),
                                value=True,
                                interactive=True,
                                show_label=True,
                            )
                            if_save_every_weights1Bb = gr.Checkbox(
                                label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                                value=True,
                                interactive=True,
                                show_label=True,
                            )
                    with gr.Row(equal_height=True):
                        # with gr.Column():
                        save_every_epoch1Bb = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label=i18n("保存频率save_every_epoch"),
                            value=5,
                            interactive=True,
                        )
                        # with gr.Column():
                        if_dpo = gr.Checkbox(
                            label=i18n("是否开启DPO训练选项(实验性)"),
                            value=False,
                            interactive=True,
                            show_label=True,
                        )
                        with gr.Column(scale=2):
                            gpu_numbers1Bb = gr.Dropdown(
                                label=i18n("GPU卡号"),
                                choices=GPU_INDEX_LIST,
                                value=GPU_INDEX_LIST,
                                interactive=True,
                                multiselect=True,
                                allow_custom_value=False,
                            )
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            with gr.Row(equal_height=True):
                                button1Bb_open = gr.Button(
                                    value=process_info(process_name_gpt, "open"), variant="primary", visible=True
                                )
                                button1Bb_close = gr.Button(
                                    value=process_info(process_name_gpt, "close"), variant="primary", visible=False
                                )
                        with gr.Column():
                            info1Bb = gr.Textbox(label=process_info(process_name_gpt, "info"))

            button1Ba_close.click(close1Ba, [], [info1Ba, button1Ba_open, button1Ba_close])
            button1Bb_close.click(close1Bb, [], [info1Bb, button1Bb_open, button1Bb_close])

            with gr.TabItem("1C-" + i18n("推理")):
                gr.Markdown(
                    value=i18n(
                        "选择训练完存放在SoVITS_weights和GPT_weights下的模型. 默认的几个是底模, 体验5秒Zero Shot TTS不训练推理用."
                    )
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                GPT_dropdown = gr.Dropdown(
                                    label=i18n("GPT模型列表"),
                                    choices=GPT_names,
                                    value=GPT_names[0][-1],
                                    interactive=True,
                                )
                            with gr.Column():
                                SoVITS_dropdown = gr.Dropdown(
                                    label=i18n("SoVITS模型列表"),
                                    choices=SoVITS_names,
                                    value=SoVITS_names[0][-1],
                                    interactive=True,
                                )
                    with gr.Column(scale=2):
                        with gr.Row(equal_height=True):
                            gpu_number_1C = gr.Dropdown(
                                label=i18n("GPU卡号"),
                                choices=GPU_INDEX_LIST,
                                value=infer_device.index,
                                interactive=True,
                                multiselect=False,
                                allow_custom_value=False,
                            )
                            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
                    refresh_button.click(fn=change_choice, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
                with gr.Row(equal_height=True):
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            batched_infer_enabled = gr.Checkbox(
                                label=i18n("启用并行推理版本"), value=False, interactive=True, show_label=True
                            )
                        with gr.Column():
                            backends_dropdown = gr.Dropdown(
                                choices=backends_gradio,
                                label=i18n("推理后端"),
                                value=backends_gradio[-1][-1],
                                interactive=True,
                            )
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            quantization_methods_dropdown = gr.Dropdown(
                                choices=cast(list, changeQuantization(backends_gradio[-1][-1], gradio_call=False)),
                                value="None",
                                label=i18n("量化方法"),
                                interactive=True,
                            )
                        open_tts = gr.Button(
                            value=process_info(process_name_tts, "open"), variant="primary", visible=True
                        )
                        close_tts = gr.Button(
                            value=process_info(process_name_tts, "close"), variant="primary", visible=False
                        )

                    backends_dropdown.change(
                        changeQuantization,
                        [backends_dropdown],
                        [quantization_methods_dropdown],
                    )

                    open_tts.click(
                        change_tts_inference,
                        [
                            gpu_number_1C,
                            GPT_dropdown,
                            SoVITS_dropdown,
                            batched_infer_enabled,
                            backends_dropdown,
                            quantization_methods_dropdown,
                        ],
                        [open_tts, close_tts, batched_infer_enabled],
                    )
                    close_tts.click(
                        change_tts_inference,
                        [
                            gpu_number_1C,
                            GPT_dropdown,
                            SoVITS_dropdown,
                            batched_infer_enabled,
                            backends_dropdown,
                            quantization_methods_dropdown,
                        ],
                        [open_tts, close_tts, batched_infer_enabled],
                    )
            button1Ba_open.click(
                open1Ba,
                [
                    version_checkbox,
                    batch_size,
                    total_epoch,
                    exp_name,
                    text_low_lr_rate,
                    if_save_latest,
                    if_save_every_weights,
                    save_every_epoch,
                    gpu_numbers1Ba,
                    pretrained_s2G,
                    pretrained_s2D,
                    if_grad_ckpt,
                    lora_rank,
                ],
                [info1Ba, button1Ba_open, button1Ba_close, SoVITS_dropdown, GPT_dropdown],
            )
            button1Bb_open.click(
                open1Bb,
                [
                    batch_size1Bb,
                    total_epoch1Bb,
                    exp_name,
                    if_dpo,
                    if_save_latest1Bb,
                    if_save_every_weights1Bb,
                    save_every_epoch1Bb,
                    gpu_numbers1Bb,
                    pretrained_s1,
                ],
                [info1Bb, button1Bb_open, button1Bb_close, SoVITS_dropdown, GPT_dropdown],
            )
            version_checkbox.change(
                switch_version,
                [version_checkbox],
                [
                    pretrained_s2G,
                    pretrained_s2D,
                    pretrained_s1,
                    GPT_dropdown,
                    SoVITS_dropdown,
                    batch_size,
                    total_epoch,
                    save_every_epoch,
                    text_low_lr_rate,
                    if_grad_ckpt,
                    batched_infer_enabled,
                    lora_rank,
                ],
            )

        with gr.TabItem(i18n("2-GPT-SoVITS-变声")):
            gr.Markdown(value=i18n("施工中, 请静候佳音"))

app.queue().launch(
    server_name="0.0.0.0",
    inbrowser=True,
    share=is_share,
    server_port=webui_port_main,
)
