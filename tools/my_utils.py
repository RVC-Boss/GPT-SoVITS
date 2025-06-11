import ctypes
import os
import sys
from pathlib import Path

import ffmpeg
import gradio as gr
import numpy as np
import pandas as pd

from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language=os.environ.get("language", "Auto"))


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) is False:
            raise RuntimeError("You input a wrong audio path that does not exists, please fix it!")
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True)
        )  # Expose the Error
        raise RuntimeError(i18n("音频加载失败"))

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str: str):
    if path_str.endswith(("\\", "/")):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace("/", os.sep).replace("\\", os.sep)
    return path_str.strip(
        " '\n\"\u202a"
    )  # path_str.strip(" ").strip('\'').strip("\n").strip('"').strip(" ").strip("\u202a")


def check_for_existance(file_list: list = None, is_train=False, is_dataset_processing=False):
    files_status = []
    if is_train == True and file_list:
        file_list.append(os.path.join(file_list[0], "2-name2text.txt"))
        file_list.append(os.path.join(file_list[0], "3-bert"))
        file_list.append(os.path.join(file_list[0], "4-cnhubert"))
        file_list.append(os.path.join(file_list[0], "5-wav32k"))
        file_list.append(os.path.join(file_list[0], "6-name2semantic.tsv"))
    for file in file_list:
        if os.path.exists(file):
            files_status.append(True)
        else:
            files_status.append(False)
    if sum(files_status) != len(files_status):
        if is_train:
            for file, status in zip(file_list, files_status):
                if status:
                    pass
                else:
                    gr.Warning(file)
            gr.Warning(i18n("以下文件或文件夹不存在"))
            return False
        elif is_dataset_processing:
            if files_status[0]:
                return True
            elif not files_status[0]:
                gr.Warning(file_list[0])
            elif not files_status[1] and file_list[1]:
                gr.Warning(file_list[1])
            gr.Warning(i18n("以下文件或文件夹不存在"))
            return False
        else:
            if file_list[0]:
                gr.Warning(file_list[0])
                gr.Warning(i18n("以下文件或文件夹不存在"))
            else:
                gr.Warning(i18n("路径不能为空"))
            return False
    return True


def check_details(path_list=None, is_train=False, is_dataset_processing=False):
    if is_dataset_processing:
        list_path, audio_path = path_list
        if not list_path.endswith(".list"):
            gr.Warning(i18n("请填入正确的List路径"))
            return
        if audio_path:
            if not os.path.isdir(audio_path):
                gr.Warning(i18n("请填入正确的音频文件夹路径"))
                return
        with open(list_path, "r", encoding="utf8") as f:
            line = f.readline().strip("\n").split("\n")
        wav_name, _, __, ___ = line[0].split("|")
        wav_name = clean_path(wav_name)
        if audio_path != "" and audio_path != None:
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s" % (audio_path, wav_name)
        else:
            wav_path = wav_name
        if os.path.exists(wav_path):
            ...
        else:
            gr.Warning(wav_path + i18n("路径错误"))
        return
    if is_train:
        path_list.append(os.path.join(path_list[0], "2-name2text.txt"))
        path_list.append(os.path.join(path_list[0], "4-cnhubert"))
        path_list.append(os.path.join(path_list[0], "5-wav32k"))
        path_list.append(os.path.join(path_list[0], "6-name2semantic.tsv"))
        phone_path, hubert_path, wav_path, semantic_path = path_list[1:]
        with open(phone_path, "r", encoding="utf-8") as f:
            if f.read(1):
                ...
            else:
                gr.Warning(i18n("缺少音素数据集"))
        if os.listdir(hubert_path):
            ...
        else:
            gr.Warning(i18n("缺少Hubert数据集"))
        if os.listdir(wav_path):
            ...
        else:
            gr.Warning(i18n("缺少音频数据集"))
        df = pd.read_csv(semantic_path, delimiter="\t", encoding="utf-8")
        if len(df) >= 1:
            ...
        else:
            gr.Warning(i18n("缺少语义数据集"))


def load_cudnn():
    import torch

    if not torch.cuda.is_available():
        print("[INFO] CUDA is not available, skipping cuDNN setup.")
        return

    if sys.platform == "win32":
        torch_lib_dir = Path(torch.__file__).parent / "lib"
        if torch_lib_dir.exists():
            os.add_dll_directory(str(torch_lib_dir))
            print(f"[INFO] Added DLL directory: {torch_lib_dir}")
            matching_files = sorted(torch_lib_dir.glob("cudnn_cnn*.dll"))
            if not matching_files:
                print(f"[ERROR] No cudnn_cnn*.dll found in {torch_lib_dir}")
                return
            for dll_path in matching_files:
                dll_name = os.path.basename(dll_path)
                try:
                    ctypes.CDLL(dll_name)
                    print(f"[INFO] Loaded: {dll_name}")
                except OSError as e:
                    print(f"[WARNING] Failed to load {dll_name}: {e}")
        else:
            print(f"[WARNING] Torch lib directory not found: {torch_lib_dir}")

    elif sys.platform == "linux":
        site_packages = Path(torch.__file__).resolve().parents[1]
        cudnn_dir = site_packages / "nvidia" / "cudnn" / "lib"

        if not cudnn_dir.exists():
            print(f"[ERROR] cudnn dir not found: {cudnn_dir}")
            return

        matching_files = sorted(cudnn_dir.glob("libcudnn_cnn*.so*"))
        if not matching_files:
            print(f"[ERROR] No libcudnn_cnn*.so* found in {cudnn_dir}")
            return

        for so_path in matching_files:
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)  # type: ignore
                print(f"[INFO] Loaded: {so_path}")
            except OSError as e:
                print(f"[WARNING] Failed to load {so_path}: {e}")


def load_nvrtc():
    import torch

    if not torch.cuda.is_available():
        print("[INFO] CUDA is not available, skipping nvrtc setup.")
        return

    if sys.platform == "win32":
        torch_lib_dir = Path(torch.__file__).parent / "lib"
        if torch_lib_dir.exists():
            os.add_dll_directory(str(torch_lib_dir))
            print(f"[INFO] Added DLL directory: {torch_lib_dir}")
            matching_files = sorted(torch_lib_dir.glob("nvrtc*.dll"))
            if not matching_files:
                print(f"[ERROR] No nvrtc*.dll found in {torch_lib_dir}")
                return
            for dll_path in matching_files:
                dll_name = os.path.basename(dll_path)
                try:
                    ctypes.CDLL(dll_name)
                    print(f"[INFO] Loaded: {dll_name}")
                except OSError as e:
                    print(f"[WARNING] Failed to load {dll_name}: {e}")
        else:
            print(f"[WARNING] Torch lib directory not found: {torch_lib_dir}")

    elif sys.platform == "linux":
        site_packages = Path(torch.__file__).resolve().parents[1]
        nvrtc_dir = site_packages / "nvidia" / "cuda_nvrtc" / "lib"

        if not nvrtc_dir.exists():
            print(f"[ERROR] nvrtc dir not found: {nvrtc_dir}")
            return

        matching_files = sorted(nvrtc_dir.glob("libnvrtc*.so*"))
        if not matching_files:
            print(f"[ERROR] No libnvrtc*.so* found in {nvrtc_dir}")
            return

        for so_path in matching_files:
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)  # type: ignore
                print(f"[INFO] Loaded: {so_path}")
            except OSError as e:
                print(f"[WARNING] Failed to load {so_path}: {e}")
