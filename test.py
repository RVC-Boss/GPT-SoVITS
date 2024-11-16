import requests

# API configuration
host = '127.0.0.1'
port = 9880
url = f'http://{host}:{port}/tts'

# Parameters for the request
params = {
    'text': 'ははは、それはいいよね！でもなんか対処法ではないよなこれ対処法ではないけどそもそもの話みたいなことを言ってんのか',
    'text_lang': 'ja',
    'ref_audio_path': 'test.wav',
    'prompt_lang': 'ja',
    'prompt_text': 'でもなんか対処法ではないよなこれ対処法ではないけどそもそもの話みたいなことを言ってんのか',
    'text_split_method': 'cut0',
    'batch_size': 1,
    'media_type': 'wav',
    'streaming_mode': False,
}

try:
    # Send the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the audio content to a file
        with open('output.wav', 'wb') as f:
            f.write(response.content)
        print('Audio saved to output.wav')
    else:
        print(f'Error: {response.status_code}')
        print(response.json())
except requests.exceptions.RequestException as e:
    print(f'An error occurred: {e}')
