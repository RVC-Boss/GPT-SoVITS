import os
import requests
import itertools
import Ref_Audio_Selector.config.config_manager as config_manager
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote

config = config_manager.get_config()


class URLComposer:
    def __init__(self, base_url, emotion_param_name, text_param_name, ref_path_param_name, ref_text_param_name):
        self.base_url = safe_encode_query_params(base_url)
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
            self.text_param_name: quote(text_value),
            self.emotion_param_name: quote(emotion_value),
        }
        return self._append_params_to_url(params)

    def build_url_with_ref(self, text_value, ref_path_value, ref_text_value):
        if self.emotion_param_name:
            raise ValueError("Cannot use reference parameters when emotion parameter is set.")
        params = {
            self.text_param_name: quote(text_value),
            self.ref_path_param_name: quote(ref_path_value),
            self.ref_text_param_name: quote(ref_text_value),
        }
        return self._append_params_to_url(params)

    def _append_params_to_url(self, params: dict):
        url_with_params = self.base_url
        if params:
            query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
            url_with_params += '?' + query_params if '?' not in self.base_url else '&' + query_params
        return url_with_params


def safe_encode_query_params(original_url):

    # 分析URL以获取查询字符串部分
    parsed_url = urlparse(original_url)
    query_params = parse_qs(parsed_url.query)

    # 将查询参数转换为编码过的字典（键值对会被转码）
    encoded_params = {k: quote(v[0]) for k, v in query_params.items()}

    # 重新编码查询字符串
    new_query_string = urlencode(encoded_params, doseq=False)

    # 重建完整的URL
    new_parsed_url = parsed_url._replace(query=new_query_string)
    encoded_url = urlunparse(new_parsed_url)

    print(encoded_url)
    return encoded_url


def generate_audio_files(url_composer, text_list, emotion_list, output_dir_path):
    # Ensure the output directory exists
    output_dir = os.path.abspath(output_dir_path)
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for text and emotion categories
    text_subdir = os.path.join(output_dir, config.get_inference('inference_audio_text_aggregation_dir'))
    os.makedirs(text_subdir, exist_ok=True)
    emotion_subdir = os.path.join(output_dir, config.get_inference('inference_audio_emotion_aggregation_dir'))
    os.makedirs(emotion_subdir, exist_ok=True)

    # 计算笛卡尔积
    cartesian_product = list(itertools.product(text_list, emotion_list))

    for text, emotion in cartesian_product:
        # Generate audio byte stream using the create_audio function

        if url_composer.is_emotion():
            real_url = url_composer.build_url_with_emotion(text, emotion['emotion'])
        else:
            real_url = url_composer.build_url_with_ref(text, emotion['ref_path'], emotion['ref_text'])

        audio_bytes = inference_audio_from_api(real_url)

        emotion_name = emotion['emotion']

        text_subdir_text = os.path.join(text_subdir, text)
        os.makedirs(text_subdir_text, exist_ok=True)
        text_subdir_text_file_path = os.path.join(text_subdir_text, emotion_name + '.wav')

        emotion_subdir_emotion = os.path.join(emotion_subdir, emotion_name)
        os.makedirs(emotion_subdir_emotion, exist_ok=True)
        emotion_subdir_emotion_file_path = os.path.join(emotion_subdir_emotion, text + '.wav')

        # Write audio bytes to the respective files
        with open(text_subdir_text_file_path, 'wb') as f:
            f.write(audio_bytes)
        with open(emotion_subdir_emotion_file_path, 'wb') as f:
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
