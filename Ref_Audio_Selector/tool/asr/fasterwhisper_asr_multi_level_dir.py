import argparse
import os
import traceback
import Ref_Audio_Selector.config_param.config_params as params

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

from tools.asr.config import check_fw_local_models
from Ref_Audio_Selector.config_param.log_config import logger

language_code_list = [
    "af", "am", "ar", "as", "az",
    "ba", "be", "bg", "bn", "bo",
    "br", "bs", "ca", "cs", "cy",
    "da", "de", "el", "en", "es",
    "et", "eu", "fa", "fi", "fo",
    "fr", "gl", "gu", "ha", "haw",
    "he", "hi", "hr", "ht", "hu",
    "hy", "id", "is", "it", "ja",
    "jw", "ka", "kk", "km", "kn",
    "ko", "la", "lb", "ln", "lo",
    "lt", "lv", "mg", "mi", "mk",
    "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no",
    "oc", "pa", "pl", "ps", "pt",
    "ro", "ru", "sa", "sd", "si",
    "sk", "sl", "sn", "so", "sq",
    "sr", "su", "sv", "sw", "ta",
    "te", "tg", "th", "tk", "tl",
    "tr", "tt", "uk", "ur", "uz",
    "vi", "yi", "yo", "zh", "yue",
    "auto"]


def execute_asr_multi_level_dir(input_folder, output_folder, model_size, language, precision):
    if '-local' in model_size:
        model_size = model_size[:-6]
        model_path = f'tools/asr/models/faster-whisper-{model_size}'
    else:
        model_path = model_size
    if language == 'auto':
        language = None  # 不设置语种由模型自动输出概率最高的语种
    logger.info("loading faster whisper model:", model_size, model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = WhisperModel(model_path, device=device, compute_type=precision)
    except:
        return logger.error(traceback.format_exc())

    output = []

    # 递归遍历输入目录及所有子目录
    for root, dirs, files in os.walk(input_folder):
        for file_name in sorted(files):
            # 只处理wav文件（假设是wav文件）
            if file_name.endswith(".wav"):
                try:
                    file_path = os.path.join(root, file_name)
                    original_text = os.path.basename(root)
                    segments, info = model.transcribe(
                        audio=file_path,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=700),
                        language=language)
                    text = ''

                    if info.language == "zh":
                        logger.info("检测为中文文本, 转 FunASR 处理")
                        if ("only_asr" not in globals()):
                            from Ref_Audio_Selector.tool.asr.funasr_asr_multi_level_dir import \
                                only_asr  # #如果用英文就不需要导入下载模型
                        text = only_asr(file_path)

                    if text == '':
                        for segment in segments:
                            text += segment.text
                    output.append(f"{file_path}|{original_text}|{info.language.upper()}|{text}")
                    print(f"{file_path}|{original_text}|{info.language.upper()}|{text}")
                except:
                    return logger.error(traceback.format_exc())

    output_folder = output_folder
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(f'{output_folder}/{params.asr_filename}.list')

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        logger.info(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
    return output_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="Path to the folder containing WAV files.")
    parser.add_argument("-o", "--output_folder", type=str, required=True,
                        help="Output folder to store transcriptions.")
    parser.add_argument("-s", "--model_size", type=str, default='large-v3',
                        choices=check_fw_local_models(),
                        help="Model Size of Faster Whisper")
    parser.add_argument("-l", "--language", type=str, default='ja',
                        choices=language_code_list,
                        help="Language of the audio files.")
    parser.add_argument("-p", "--precision", type=str, default='float16', choices=['float16', 'float32'],
                        help="fp16 or fp32")

    cmd = parser.parse_args()
    output_file_path = execute_asr_multi_level_dir(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        model_size=cmd.model_size,
        language=cmd.language,
        precision=cmd.precision,
    )
