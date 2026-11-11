# -*- coding:utf-8 -*-

import argparse
import os
import sys
import traceback

import setuptools._distutils as _distutils
import setuptools._distutils.util as _distutils_util
import setuptools._distutils.version as _distutils_version

sys.modules.setdefault("distutils", _distutils)
sys.modules.setdefault("distutils.util", _distutils_util)
sys.modules.setdefault("distutils.version", _distutils_version)

from funasr import AutoModel
from modelscope import snapshot_download
from tqdm import tqdm

funasr_models = {}  # 存储模型避免重复加载
ASR_MODEL_ROOT = "tools/asr/models"
FUN_ASR_NANO_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
PARAFORMER_MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
UNIASR_MODEL_ID = "iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online"
FUN_ASR_NANO_MODEL_PATH = f"{ASR_MODEL_ROOT}/Fun-ASR-Nano-2512"
VAD_MODEL_PATH = f"{ASR_MODEL_ROOT}/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL_PATH = f"{ASR_MODEL_ROOT}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
PARAFORMER_MODEL_PATH = f"{ASR_MODEL_ROOT}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
UNIASR_MODEL_PATH = f"{ASR_MODEL_ROOT}/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online"


def ensure_model_dir(model_id, model_path, revision=None):
    if os.path.isfile(os.path.join(model_path, "model.pt")):
        return model_path

    download_kwargs = {"local_dir": model_path}
    if revision is not None:
        download_kwargs["revision"] = revision
    print(f"Downloading FunASR model: {model_id} -> {model_path}")
    snapshot_download(model_id, **download_kwargs)
    return model_path


def only_asr(input_file, language, backend="fun-asr-nano"):
    try:
        model = create_model(language, backend=backend)
        text = model.generate(input=input_file)[0]["text"]
    except Exception:
        text = ""
        print(traceback.format_exc())
    return text


def disable_flash_attention_probe():
    import transformers.utils
    import transformers.utils.import_utils

    transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
    transformers.utils.is_flash_attn_2_available = lambda: False


def create_fun_asr_nano_model(device):
    disable_flash_attention_probe()
    ensure_model_dir(FUN_ASR_NANO_MODEL_ID, FUN_ASR_NANO_MODEL_PATH)
    ensure_model_dir(VAD_MODEL_ID, VAD_MODEL_PATH)
    return AutoModel(
        model=FUN_ASR_NANO_MODEL_PATH,
        vad_model=VAD_MODEL_PATH,
        device=device,
        disable_update=True,
        check_latest=False,
        vad_kwargs={"check_latest": False},
        hub="ms",
    )


def create_damo_model(device):
    ensure_model_dir(VAD_MODEL_ID, VAD_MODEL_PATH, revision="v2.0.4")
    ensure_model_dir(PUNC_MODEL_ID, PUNC_MODEL_PATH, revision="v2.0.4")
    ensure_model_dir(PARAFORMER_MODEL_ID, PARAFORMER_MODEL_PATH, revision="v2.0.4")
    return AutoModel(
        model=PARAFORMER_MODEL_PATH,
        model_revision="v2.0.4",
        vad_model=VAD_MODEL_PATH,
        vad_model_revision="v2.0.4",
        vad_kwargs={"check_latest": False},
        punc_model=PUNC_MODEL_PATH,
        punc_model_revision="v2.0.4",
        punc_kwargs={"check_latest": False},
        device=device,
        disable_update=True,
        check_latest=False,
        hub="ms",
    )


def create_uniasr_model(device):
    ensure_model_dir(UNIASR_MODEL_ID, UNIASR_MODEL_PATH, revision="master")
    return AutoModel(
        model=UNIASR_MODEL_PATH,
        model_revision="master",
        device=device,
        disable_update=True,
        check_latest=False,
        hub="ms",
    )


def create_model(language="zh", **kwargs):
    backend = kwargs.get("backend", "fun-asr-nano")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = f"{backend}_{language}"
    if cache_key in funasr_models:
        return funasr_models[cache_key]

    if backend == "damo":
        if language != "zh":
            raise ValueError("达摩 ASR 仅支持 zh")
        model = create_damo_model(device)
        print(f"FunASR 达摩 Paraformer 模型加载完成: {language.upper()}")
    elif backend == "fun-asr-nano" and language == "yue":
        model = create_uniasr_model(device)
        print(f"FunASR UniASR 粤语模型加载完成: {language.upper()}")
    elif backend == "fun-asr-nano":
        model = create_fun_asr_nano_model(device)
        print(f"FunASR Fun-ASR-Nano 模型加载完成: {language.upper()}")
    else:
        raise ValueError(f"{backend} is not supported")

    funasr_models[cache_key] = model
    return model


def execute_asr(input_folder, output_folder, model_size, language, backend="fun-asr-nano"):
    input_file_names = os.listdir(input_folder)
    input_file_names.sort()

    output = []
    output_file_name = os.path.basename(input_folder)

    model = create_model(language, backend=backend)

    for file_name in tqdm(input_file_names):
        try:
            print("\n" + file_name)
            file_path = os.path.join(input_folder, file_name)
            text = model.generate(input=file_path)[0]["text"]
            output.append(f"{file_path}|{output_file_name}|{language.upper()}|{text}")
        except Exception:
            print(traceback.format_exc())

    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(f"{output_folder}/{output_file_name}.list")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        print(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
    return output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_folder", type=str, required=True, help="Path to the folder containing WAV files."
    )
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output folder to store transcriptions.")
    parser.add_argument("-s", "--model_size", type=str, default="large", help="Model Size of FunASR is Large")
    parser.add_argument(
        "-l", "--language", type=str, default="zh", choices=["zh", "yue", "ja", "en", "ko", "auto"], help="Language of the audio files."
    )
    parser.add_argument(
        "-p", "--precision", type=str, default="float16", choices=["float16", "float32"], help="fp16 or fp32"
    )  # 还没接入
    parser.add_argument(
        "--backend", type=str, default="fun-asr-nano", choices=["fun-asr-nano", "damo"], help="ASR backend"
    )
    cmd = parser.parse_args()
    execute_asr(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        model_size=cmd.model_size,
        language=cmd.language,
        backend=cmd.backend,
    )
