import requests
import wave
import io
import sounddevice as sd

# API configuration
host = '127.0.0.1'
port = 9880
url = f'http://{host}:{port}/tts'

# Parameters for the request
params = {
    'text': 'ふふふ、それはいいよね！でもなんか対処法ではないよなこれ対処法ではないけどそもそもの話みたいなことを言ってんのか',
    'text_lang': 'ja',
    'ref_audio_path': 'test.wav',
    'prompt_lang': 'ja',
    'prompt_text': 'でもなんか対処法ではないよなこれ対処法ではないけどそもそもの話みたいなことを言ってんのか',
    'text_split_method': 'cut0',
    'batch_size': 1,
    'media_type': 'wav',
    'streaming_mode': True, 
}

while True:
    input("Waiting for enter")
    try:
        # Send the GET request with streaming enabled
        response = requests.get(url, params=params, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            buffer = b''  # Buffer to hold data until header is processed
            header_size = 44  # Standard WAV header size
            header_parsed = False
            stream = None

            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    if not header_parsed:
                        buffer += chunk
                        if len(buffer) >= header_size:
                            # Parse WAV header
                            wav_header = buffer[:header_size]
                            wav_file = wave.open(io.BytesIO(wav_header), 'rb')
                            channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()
                            sample_rate = wav_file.getframerate()
                            wav_file.close()

                            # Map sample_width to dtype
                            dtype_map = {1: 'int8', 2: 'int16', 3: 'int24', 4: 'int32'}
                            dtype = dtype_map.get(sample_width, 'int16')

                            # Open audio stream
                            stream = sd.RawOutputStream(
                                samplerate=sample_rate,
                                channels=channels,
                                dtype=dtype,
                                blocksize=0,  # Use default block size
                            )

                            stream.start()

                            # Write any remaining data after the header
                            data = buffer[header_size:]
                            if data:
                                stream.write(data)
                            header_parsed = True
                            buffer = b''  # Clear buffer
                    else:
                        # Play audio data
                        if stream:
                            stream.write(chunk)
            # Clean up
            if stream:
                stream.stop()
                stream.close()
            print('Audio playback completed.')
        else:
            print(f'Error: {response.status_code}')
            # Print the error message from the API
            try:
                print(response.json())
            except ValueError:
                print(response.text)
    except requests.exceptions.RequestException as e:
        print(f'An error occurred: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')
