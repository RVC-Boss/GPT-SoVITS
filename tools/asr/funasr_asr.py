# -*- coding:utf-8 -*-

import argparse
import os
import traceback
from tqdm import tqdm

from funasr import AutoModel

path_asr  = 'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
path_vad  = 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
path_punc = 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
path_asr  = path_asr  if os.path.exists(path_asr)  else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
path_vad  = path_vad  if os.path.exists(path_vad)  else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_punc = path_punc if os.path.exists(path_punc) else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

model = AutoModel(
    model               = path_asr,
    model_revision      = "v2.0.4",
    vad_model           = path_vad,
    vad_model_revision  = "v2.0.4",
    punc_model          = path_punc,
    punc_model_revision = "v2.0.4",
)

def only_asr(input_file):
    try:
        text = model.generate(input=input_file)[0]["text"]
    except:
        text = ''
        print(traceback.format_exc())
    return text

def execute_asr(input_folder, output_folder, model_size, language):
    input_file_names = os.listdir(input_folder)
    input_file_names.sort()
    
    output = []
    output_file_name = os.path.basename(input_folder)

    for file_name in tqdm(input_file_names):
        try:
            file_path = os.path.join(input_folder, file_name)
            text = model.generate(input=file_path)[0]["text"]
            output.append(f"{file_path}|{output_file_name}|{language.upper()}|{text}")
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
    parser.add_argument("-s", "--model_size", type=str, default='large',
                        help="Model Size of FunASR is Large")
    parser.add_argument("-l", "--language", type=str, default='zh', choices=['zh'],
                        help="Language of the audio files.")
    parser.add_argument("-p", "--precision", type=str, default='float16', choices=['float16','float32'],
                        help="fp16 or fp32")#还没接入

    cmd = parser.parse_args()
    execute_asr(
        input_folder  = cmd.input_folder,
        output_folder = cmd.output_folder,
        model_size    = cmd.model_size,
        language      = cmd.language,
    )
