import argparse
import os
import re
import traceback
import torch
from tqdm import tqdm
from funasr import AutoModel

model_dir = "tools/asr/models/SenseVoiceSmall"
model_dir  = model_dir if os.path.exists(model_dir)  else "iic/SenseVoiceSmall"

def execute_asr(input_folder, output_folder, language, device):
    try:
        model = AutoModel(model=model_dir,
                  vad_model="fsmn-vad",
                  vad_kwargs={"max_single_segment_time": 30000},
                )
    except:
        return print(traceback.format_exc())
    
    input_file_names = os.listdir(input_folder)
    input_file_names.sort()
    output = []
    output_file_name = os.path.basename(input_folder)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for file_name in tqdm(input_file_names):
        try:
            file_path = os.path.join(input_folder, file_name)
            res = model.generate(
                input=file_path,
                cache={},
                language=language.lower(), # "zn", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                batch_size_s=0, 
                device = device
            )[0]['text']
            text_language = re.search(r'<([^<>]+)>', res).group(1)[1:-1].upper() if language == 'auto' else language.upper()
            text = re.sub(r'<[^<>]*>', '', res).replace(' ', '')
            output.append(f"{file_path}|{output_file_name}|{text_language}|{text}")
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
    parser.add_argument("-l", "--language", type=str, default='auto',
                        choices=['auto','zh','en','ja'],
                        help="Language of the audio files.")
    parser.add_argument("-d", "--device", type=str, default=None, choices=['cpu','cuda'],
                        help="CPU or CUDA")

    cmd = parser.parse_args()
    output_file_path = execute_asr(
        input_folder  = cmd.input_folder,
        output_folder = cmd.output_folder,
        language      = cmd.language,
        device        = cmd.device,
    )
