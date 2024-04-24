import os
import requests
import urllib.parse

class URLComposer:
    def __init__(self, base_url, emotion_param_name, text_param_name, ref_path_param_name, ref_text_param_name):
        self.base_url = base_url
        self.emotion_param_name = emotion_param_name
        self.text_param_name = text_param_name
        self.ref_path_param_name = ref_path_param_name
        self.ref_text_param_name = ref_text_param_name
        
    
    def is_valid(self):
        if self.base_url is None or self.base_url == '':
            raise ValueError("请输入url")
        
        if self.text_param_name is None or self.text_param_name == '':
            raise ValueError("请输入text参数名")
        
        if self.emotion_param_name is None and self.ref_path_param_name is None and self.ref_text_param_name is None:
            raise ValueError("请输入至少一个参考or情绪的参数")
        
    def is_emotion(self):
        return self.emotion_param_name is not None and self.emotion_param_name != ''

    def build_url_with_emotion(self, text_value, emotion_value):
        if not self.emotion_param_name:
            raise ValueError("Emotion parameter name is not set.")
        params = {
            self.text_param_name: urllib.parse.quote(text_value),
            self.emotion_param_name: urllib.parse.quote(emotion_value),
        }
        return self._append_params_to_url(params)

    def build_url_with_ref(self, text_value, ref_path_value, ref_text_value):
        if self.emotion_param_name:
            raise ValueError("Cannot use reference parameters when emotion parameter is set.")
        params = {
            self.text_param_name: urllib.parse.quote(text_value),
            self.ref_path_param_name: urllib.parse.quote(ref_path_value),
            self.ref_text_param_name: urllib.parse.quote(ref_text_value),
        }
        return self._append_params_to_url(params)

    def _append_params_to_url(self, params: dict):
        url_with_params = self.base_url
        if params:
            query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
            url_with_params += '?' + query_params if '?' not in self.base_url else '&' + query_params
        return url_with_params
    
    
def generate_audio_files(url_composer, text_list, emotion_list, output_dir_path):

    # Ensure the output directory exists
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for text and emotion categories
    text_subdir = os.path.join(output_dir, 'text')
    text_subdir.mkdir(exist_ok=True)
    emotion_subdir = os.path.join(output_dir, 'emotion')
    emotion_subdir.mkdir(exist_ok=True)

    for text, emotion in zip(text_list, emotion_list):
        # Generate audio byte stream using the create_audio function
        
        if url_composer.is_emotion():
            real_url = url_composer.build_url_with_emotion(text, emotion['emotion'])
        else:
            real_url = url_composer.build_url_with_ref(text, emotion['ref_path'], emotion['ref_text'])
        
        audio_bytes = inference_audio_from_api(real_url)

        emotion_name = emotion['emotion']

        # Save audio files in both directories with the desired structure
        text_file_path = os.path.join(text_subdir, text, emotion_name, '.wav')
        emotion_file_path = os.path.join(emotion_subdir, emotion_name, text, '.wav')

        # Ensure intermediate directories for nested file paths exist
        text_file_path.parent.mkdir(parents=True, exist_ok=True)
        emotion_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write audio bytes to the respective files
        with open(text_file_path, 'wb') as f:
            f.write(audio_bytes)
        with open(emotion_file_path, 'wb') as f:
            f.write(audio_bytes)
    


def inference_audio_from_api(url):

    # 发起GET请求
    response = requests.get(url, stream=True)

    # 检查响应状态码是否正常（例如200表示成功）
    if response.status_code == 200:
        # 返回音频数据的字节流
        return response.content
    else:
        raise Exception(f"Failed to fetch audio from API. Server responded with status code {response.status_code}.")