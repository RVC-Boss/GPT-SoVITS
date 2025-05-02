'''
If you want to generate a single audio file with GPT-SoVITS, you can use this script.
The def run() function in GPT_SoVITS.TTS_infer_pack.TTS.py is used to generate the audio, it's a generator function so it must be called with a loop.
'''

import os
import sys
import queue
import threading
import numpy as np
import sounddevice as sd
import wave
import soundfile as sf

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, 'GPT_SoVITS'))
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

def main():
    # create output directory for inference outputs
    output_dir = "tts_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = 'GPT_SoVITS/configs/tts_infer.yaml' # path to the config file 
    t2s_ckpt = 'GPT_SoVITS/pretrained_models/s1v3.ckpt' # path to the t2s checkpoint
    vits_ckpt = 'GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth' # path to the vits checkpoint
    ref_audio = 'local_files/test.wav' # path to the reference audio file
    prompt_text = 'Flesh rots, carrion feeds the scavengers, and the bones remain.  All part of the cycle of life.' # prompt text
    text = "Hey there! This is a test of the TTS streaming.  Is there anything that I can do to help you out? Or maybe you'd just like a quick snack...  If not, that's okay too.  I'm just here to chat and maybe become friends as I don't meet many people in this world." # text to be converted to audio   
    seed = 1 # -1 is random seed
    
    cfg = TTS_Config(config_path)
    pipeline = TTS(cfg)
    pipeline.init_t2s_weights(t2s_ckpt)
    pipeline.init_vits_weights(vits_ckpt)

    inputs = {
        "text": text,
        "text_lang": "en",
        "ref_audio_path": ref_audio,
        "prompt_text": prompt_text,
        "prompt_lang": "en",
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "sample_steps": 10,
        "seed" : seed   
    }

    while True:
        input("Enter to generate audio")
        gen = pipeline.run(inputs)
        idx = 0
        for sr, fragment in gen:
            out_path = os.path.join(output_dir, f"inference_{idx}.wav")
            while os.path.exists(out_path):
                idx += 1
                out_path = os.path.join(output_dir, f"inference_{idx}.wav")
            sf.write(out_path, fragment, sr)

if __name__ == '__main__':
    main()
