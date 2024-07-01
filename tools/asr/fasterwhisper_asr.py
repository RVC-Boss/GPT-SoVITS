import argparse
import os
import traceback

os.environ["HF_ENDPOINT"]          = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

from tools.asr.config import check_fw_local_models

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

def execute_asr(input_folder, output_folder, model_size, language, precision):
    if '-local' in model_size:
        model_size = model_size[:-6]
        model_path = f'tools/asr/models/faster-whisper-{model_size}'
    else:
        model_path = model_size
    if language == 'auto':
        language = None #不设置语种由模型自动输出概率最高的语种
    print("loading faster whisper model:",model_size,model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = WhisperModel(model_path, device=device, compute_type=precision)
    except:
        return print(traceback.format_exc())
    
    input_file_names = os.listdir(input_folder)
    input_file_names.sort()

    output = []
    output_file_name = os.path.basename(input_folder)
    
    for file_name in tqdm(input_file_names):
        try:
            file_path = os.path.join(input_folder, file_name)
            segments, info = model.transcribe(
                audio          = file_path,
                beam_size      = 5,
                vad_filter     = True,
                vad_parameters = dict(min_silence_duration_ms=700),
                language       = language)
            text = ''

            if info.language == "zh":
                print("检测为中文文本, 转 FunASR 处理")
                if("only_asr"not in globals()):
                    from tools.asr.funasr_asr import \
                        only_asr  # #如果用英文就不需要导入下载模型
                text = only_asr(file_path)

            if text == '':
                for segment in segments:
                    text += segment.text
            output.append(f"{file_path}|{output_file_name}|{info.language.upper()}|{text}")
        except:
            print(traceback.format_exc())
    
    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(f'{output_folder}/{output_file_name}.list')

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        print(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
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
    parser.add_argument("-p", "--precision", type=str, default='float16', choices=['float16','float32'],
                        help="fp16 or fp32")

    cmd = parser.parse_args()
    output_file_path = execute_asr(
        input_folder  = cmd.input_folder,
        output_folder = cmd.output_folder,
        model_size    = cmd.model_size,
        language      = cmd.language,
        precision     = cmd.precision,
    )
