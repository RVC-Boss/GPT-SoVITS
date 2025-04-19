import torch
import sounddevice as sd
import time
from queue import Queue
from threading import Thread
import os

class TTS:
    def __init__(self):
        # Replace with your checkpoints and reference audio here
        # Note: Using a venv may require updating the default paths provided here
        self.bert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.cnhuhbert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        # self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/ayaka/Ayaka-e15.ckpt"
        # self.vits_checkpoint = "GPT_SoVITS/pretrained_models/ayaka/Ayaka_e3_s1848_l32.pth"
        self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        self.vits_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        self.ref_audio = "audio/ayaka/ref_audio/10_audio.wav"

        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        
        self.config = {
            "custom": {
                "bert_base_path": self.bert_checkpoint,
                "cnhuhbert_base_path": self.cnhuhbert_checkpoint,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "is_half": True,
                "t2s_weights_path": self.t2s_checkpoint,
                "vits_weights_path": self.vits_checkpoint,
                "version": "v3"
            }
        }
        
        self.tts = TTS(TTS_Config(self.config))
        
        self.audio_queue = Queue()
        self.generating_audio = False
    
    def audio_stream(self, start_time):
        with sd.OutputStream(samplerate=32000, channels=1, dtype="int16") as stream:
            while True:
                sr, audio_data = self.audio_queue.get()
                if audio_data is None:
                    print(f"Stream Thread Done ({time.time() - start_time:.2f}s)")
                    break
                print((sr, audio_data))
                stream.write(audio_data)
            self.generating_audio = False
    
    def synthesize(self, text, start_time, generating_text=False):
        if not self.generating_audio:
            Thread(target=self.audio_stream, args=(start_time,)).start()
            self.generating_audio = True

        path = "audio/ayaka/aux_ref_audio"
        aux_ref_audios = [f"{path}/{file_name}" for file_name in os.listdir(path)]

        args = {
            "text": text,
            "text_lang": "en",
            "ref_audio_path": self.ref_audio,
            "aux_ref_audio_paths": aux_ref_audios,
            "prompt_text": "Don't worry. Now that I've experienced the event once already, I won't be easily frightened. I'll see you later. Have a lovely chat with your friend.",
            "prompt_lang": "en",
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "parallel_infer": True,
            "sample_steps": 32,
            "super_sampling": True,
            "speed_factor": 1,
            "fragment_interval": 0.2
            # "stream_output": True,
            # "max_chunk_size": 20,
        }
        
        if text:
            print(f"Synthesis Start: {time.time() - start_time}")
            generator = self.tts.run(args)
            while True:
                try:
                    audio_chunk = next(generator)
                    self.audio_queue.put(audio_chunk)
                except StopIteration:
                    break

        if not generating_text:
            self.audio_queue.put((None, None))
        
        print(f"Synthesis End ({time.time() - start_time:.2f}s)")

# Usage
tts = TTS()
"""
Time is only for debugging purposes. If not needed, feel free to remove.
Since this TTS model was built to be paired with LLM text streaming, we use a generating_text bool
this bool signifies if we are receiving the last chunk of streamed text (hence if we are generating anymore).
"""
tts.synthesize("One day, a fierce storm rolled in, bringing heavy rain and strong winds that threatened to destroy the wheat crops.", time.time(), False)
while tts.generating_audio:
    time.sleep(0.1)
tts.synthesize("One day, a fierce storm rolled in, bringing heavy rain and strong winds that threatened to destroy the wheat crops.", time.time(), False)