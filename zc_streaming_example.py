import os
import sys
import queue
import threading
import numpy as np
import sounddevice as sd
import wave
import time

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, 'GPT_SoVITS'))
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

def audio_playback_thread(audio_queue: queue.Queue, sample_rate: int):
    """
    A background thread that plays audio fragments as they become available
    in the queue using a continuous OutputStream for smooth playback.
    """
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    stream = sd.OutputStream(dtype='float32')
    stream.start()
    try:
        while True:
            audio_fragment = audio_queue.get()
            try:
                if audio_fragment is None:
                    # Sentinel received, end thread
                    break
                # ensure float32 in [-1,1]
                data = audio_fragment.astype(np.float32) / 32768.0
                stream.write(data)
            finally:
                audio_queue.task_done()
    finally:
        stream.stop()
        stream.close()
        print("Playback finished")

def main():
    config_path = 'GPT_SoVITS/configs/tts_infer.yaml' # path to the config file
    t2s_ckpt = 'GPT_SoVITS/pretrained_models/s1v3.ckpt' # path to the t2s checkpoint
    vits_ckpt = 'GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth' # path to the vits checkpoint
    ref_audio = 'local_files/test.wav' # path to the reference audio file
    prompt_text = 'Flesh rots, carrion feeds the scavengers, and the bones remain.  All part of the cycle of life.' # reference_audio transcription
    text = "Today we are going to be testing TTS streaming audio.  Is there anything that I can do to help you out? Or maybe you'd just like a quick snack...  If not, that's okay too.  I'm just here to chat and maybe become friends as I don't meet many people in this world." # text to be converted to audio
    seed = 1 # -1 is random seed
    
    # Initialize the pipeline
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
        "cumulation_amount":10,
        "search_length": 32000*2,
        "num_zeroes": 5,
        "sample_steps": 8,
        "dynamic_cumulatation": True,
        "dynamic_cumulatation_amount": 20,
        "seed" : seed
    }

    while True:
        input("enter to continue")
        fragments = []
        # Initialize generator and fetch first fragment to get sample rate
        gen = pipeline.run_generator(inputs)
        start = time.time()
        try:
            sr, fragment = next(gen)
            fragments.append(fragment)
        except StopIteration:
            print("No audio fragments generated.")
            break

        # Create audio playback queue and start thread with sample rate
        audio_queue = queue.Queue()
        playback_thread = threading.Thread(
            target=audio_playback_thread, args=(audio_queue, sr)
        )
        playback_thread.start()        

        for sr, fragment in gen:
            if len(fragments) == 1:
                audio_queue.put(fragments[0])
                end = time.time()
                print(f"Time taken to put first fragment: {end - start}")
            audio_queue.put(fragment)
            fragments.append(fragment)

        # Signal playback thread to finish and wait
        audio_queue.put(None)
        audio_queue.join()
        playback_thread.join()

        print("Audio playback complete")

if __name__ == '__main__':
    main()
