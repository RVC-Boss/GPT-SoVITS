'''
This is just an example inference script to test batching with llama, mainly for my reference in the future.
'''

import os
import sys
import numpy as np
import soundfile as sf
import threading
import queue
import sounddevice as sd
import time
import speech_recognition as sr

# Ensure that GPT_SoVITS is in the Python path
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, 'GPT_SoVITS'))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

from llama_cpp import Llama
import sys

# Initialize the Llama model
llm = Llama(
    model_path="ggml-model-q8_0.gguf",
    n_gpu_layers=-1, # Uncomment to use GPU acceleration
    seed=1337, # Uncomment to set a specific seed
    n_ctx=2048, # Uncomment to increase the context window
    chat_format="llama-3",
    verbose=False
)

from time import time

def generate_chat_completion_openai_v1_stream(messages):
    start = time()
    stream = llm.create_chat_completion_openai_v1(
        messages=messages,
        temperature=0.8,       # Adjust temperature as needed
        top_p=0.95,            # Adjust top_p as needed
        top_k=40,              # Adjust top_k as needed
        max_tokens=50,        # Adjust the maximum number of tokens as needed
        # stop=["\n"],           # Adjust the stop sequence as needed
        stream=True            # Enable streaming
    )
    end = time()
    total = end - start
    print(total)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
            
def audio_playback_thread(audio_queue, sample_rate):
    """
    Audio playback thread that plays audio fragments from the queue.
    """
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    stream = sd.OutputStream(dtype='float32')
    stream.start()

    try:
        while True:
            # Get the next audio fragment
            audio_fragment = audio_queue.get()
            try:
                if audio_fragment is None:
                    # Sentinel value received, exit the loop
                    break
                # Write the audio fragment to the stream
                stream.write(audio_fragment)
            finally:
                # Mark the item as processed
                audio_queue.task_done()
    finally:
        stream.stop()
        stream.close()

def main():
    
    config_path = 'configs/tts_infer.yaml'
    # GPT_model_path = 'pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt'
    GPT_model_path = 'custom_trained.ckpt'
    # SoVITS_model_path = 'pretrained_models/gsv-v2final-pretrained/s2G2333k.pth'
    SoVITS_model_path = 'custom_trained.pth'
    ref_audio_path = 'ref_audio.wav'
    ref_text = 'でもなんか対処法ではないよなこれ対処法ではないけどそもそもの話みたいなことを言ってんのか'
    target_text = """hahahaha, well well, let me tell you about that! it was perhaps the most exquisite day of my life! Phew, I've never had one better!  """
    output_path = 'output'
    ref_language = 'ja'
    target_language = 'ja'


    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Initialize TTS configuration and pipeline
    tts_config = TTS_Config(config_path)
    tts_pipeline = TTS(tts_config)

    # Load model weights
    tts_pipeline.init_t2s_weights(GPT_model_path)
    tts_pipeline.init_vits_weights(SoVITS_model_path)

    # Prepare inputs for TTS
    inputs = {
        "text": target_text,
        "text_lang": target_language.lower(),
        "ref_audio_path": ref_audio_path,
        "prompt_text": ref_text,
        "prompt_lang": ref_language.lower(),
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut0",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": 2855904637,
        "return_fragment": True,
        "parallel_infer": False,
        "repetition_penalty": 1.35,
    }

    # Run TTS inference

    system_message = '''You are a friendly AI named Vivy. 

    HOW YOU SHOULD RESPOND:
    - The responses should include only verbal responses, for example *laughs* should be replaced with haha
    '''

    # Initialize conversation history with system message
    conversation_history = [
        {"role": "system", "content": f"{system_message}"}
    ]
    
    # Create a queue for audio fragments
    audio_queue = queue.Queue(maxsize=100)  # Adjust maxsize based on your needs

    # Start the audio playback thread
    playback_thread = threading.Thread(
        target=audio_playback_thread,
        args=(audio_queue, tts_pipeline.configs.sampling_rate)
    )
    playback_thread.start()
    
    # Setup speech recognition
    r = sr.Recognizer()
    mic = sr.Microphone()
    
    try:
        while True:
            # Prompt for speech input instead of text input
            while True:
                print("\nPlease speak your message (say 'quit' to exit):")
                with mic as source:
                    # Adjust for ambient noise to improve recognition accuracy
                    r.adjust_for_ambient_noise(source, duration=1.0)
                    print("Listening...")
                    audio_data = r.listen(source, timeout=None, phrase_time_limit=60)
                try:
                    # Replace 'recognize_whisper' with your actual recognition method
                    # Ensure that the method is correctly implemented or available
                    user_input = r.recognize_whisper(audio_data=audio_data, model="base")
                    print("You said: " + user_input)
                    
                    # Check if the input is not empty or just whitespace
                    if user_input.strip() == "":
                        print("No speech detected. Please try again.")
                        continue  # Continue listening
                    break  # Valid input received, exit inner loop
                except sr.UnknownValueError:
                    print("Sorry, I could not understand the audio. Please try again.")
                    continue  # Continue listening
                except sr.RequestError as e:
                    print(f"Could not request results from speech recognition service; {e}")
                    continue  # Continue listening

            # Check if the user wants to quit
            if user_input.lower() == "quit":
                print("Exiting the application. Goodbye!")
                sys.exit()
            
            # Append user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Initialize variables to track character count and buffering
            buffer = ""
            char_count = 0
            waiting_for_punctuation = False
            assistant_buffer = ""

            # Generate and print the chat completion with streaming
            for token in generate_chat_completion_openai_v1_stream(conversation_history):
                print(token, end="", flush=True)  # Print each character as it's generated
                buffer += token
                assistant_buffer += token
                char_count += len(token)

                if not waiting_for_punctuation:
                    if char_count >= 100:
                        waiting_for_punctuation = True  # Start looking for punctuation
                else:
                    if any(punct in token for punct in ['.', '!', '?']):
                        # Send the buffer to TTS
                        inputs["text"] = buffer
                        synthesis_result = tts_pipeline.run_generator(inputs)
                        # Consume the generator and put audio fragments into the queue
                        for sampling_rate, audio_fragment in synthesis_result:
                            audio_queue.put(audio_fragment)
                        #put sielnce into audio queue after tts sythesis generator has finished
                        silence_duration = 0.5  # in seconds
                        num_samples = int(sampling_rate * silence_duration)
                        silence = np.zeros(num_samples, dtype='float32')
                        audio_queue.put(silence)
                        
                        # Reset counters and buffer
                        char_count = 0
                        buffer = ""
                        waiting_for_punctuation = False

            # Append assistant message to conversation history
            conversation_history.append({"role": "assistant", "content": assistant_buffer})

            # Handle any remaining text after the generator is done
            if buffer.strip():
                inputs["text"] = buffer
                synthesis_result = tts_pipeline.run_generator(inputs)

                # Consume the generator and put audio fragments into the queue
                for sampling_rate, audio_fragment in synthesis_result:
                    audio_queue.put(audio_fragment)
                #put sielnce into audio queue after tts sythesis generator has finished
                silence_duration = 0.5  # in seconds
                num_samples = int(sampling_rate * silence_duration)
                silence = np.zeros(num_samples, dtype='float32')
                audio_queue.put(silence)

                conversation_history.append({"role": "assistant", "content": buffer})
                buffer = ""
                char_count = 0
                waiting_for_punctuation = False
    finally:
        # After all processing is done, send a sentinel to the audio queue and wait for threads to finish
        audio_queue.put(None)
        audio_queue.join()
        playback_thread.join()
            
            
        # text = input("GO:")
        # inputs["text"] = text
        # synthesis_result = tts_pipeline.run_generator(inputs)
        # audio_data_list = list(synthesis_result)
        # if audio_data_list:
        #     # Since return_fragment is False, we expect only one tuple in audio_data_list
        #     sampling_rate, audio_data = audio_data_list[0]
        #     output_wav_path = os.path.join(output_path, "output.wav")
        #     # Save the audio data to a WAV file
        #     sf.write(output_wav_path, audio_data, sampling_rate)
        #     print(f"Audio saved to {output_wav_path}")
        # else:
        #     print("No audio data generated.")

if __name__ == '__main__':
    main()
