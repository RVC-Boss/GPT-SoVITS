import time
import os
import requests
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import Ref_Audio_Selector.config_param.config_params as params
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote
from Ref_Audio_Selector.config_param.log_config import logger, p_logger


class SetModelURLComposer:
    def __init__(self, type, base_url, gpt_param_name, sovits_param_name):
        self.type = type
        self.base_url = base_url
        self.gpt_param_name = gpt_param_name
        self.sovits_param_name = sovits_param_name

    def is_valid(self):
        if self.base_url is None or self.base_url == '':
            raise Exception("请求地址不能为空")
        if self.type in ['gpt', 'all']:
            if self.gpt_param_name is None or self.gpt_param_name == '':
                raise Exception("GPT参数名不能为空")
        if self.type in ['sovits', 'all']:
            if self.sovits_param_name is None or self.sovits_param_name == '':
                raise Exception("Sovits参数名不能为空")

    def build_get_url(self, value_array, need_url_encode=True):
        params = {}
        if self.type == 'gpt':
            params[self.gpt_param_name] = value_array[0]
        if self.type == 'sovits':
            params[self.sovits_param_name] = value_array[0]
        if self.type == 'all':
            params[self.gpt_param_name] = value_array[0]
            params[self.sovits_param_name] = value_array[1]
        return append_params_to_url(self.base_url, params, need_url_encode)

    def build_post_url(self, value_array, need_url_encode=True):
        url = append_params_to_url(self.base_url, {}, need_url_encode)
        params = {}
        if self.type == 'gpt':
            params[self.gpt_param_name] = value_array[0]
        if self.type == 'sovits':
            params[self.sovits_param_name] = value_array[0]
        if self.type == 'all':
            params[self.gpt_param_name] = value_array[0]
            params[self.sovits_param_name] = value_array[1]
        return url, params


class TTSURLComposer:
    def __init__(self, base_url, refer_type_param, emotion_param_name, text_param_name, ref_path_param_name, ref_text_param_name):
        self.base_url = base_url
        # 角色情绪 or 参考音频
        self.refer_type_param = refer_type_param 
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
        return self.refer_type_param == '角色情绪'

    def build_url_with_emotion(self, text_value, emotion_value, need_url_encode=True):
        params = {
            self.text_param_name: text_value,
            self.emotion_param_name: emotion_value,
        }
        return append_params_to_url(self.base_url, params, need_url_encode)

    def build_url_with_ref(self, text_value, ref_path_value, ref_text_value, need_url_encode=True):
        params = {
            self.text_param_name: text_value,
            self.ref_path_param_name: ref_path_value,
            self.ref_text_param_name: ref_text_value,
        }
        return append_params_to_url(self.base_url, params, need_url_encode)


def append_params_to_url(url_with_params, params, need_url_encode):
    if params:
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        url_with_params += '?' + query_params if '?' not in url_with_params else '&' + query_params
    return url_with_params if not need_url_encode else safe_encode_query_params(url_with_params)


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

    logger.info(encoded_url)
    return encoded_url


def generate_audio_files_parallel(url_composer, text_list, emotion_list, output_dir_path, num_processes=1):

    # 将emotion_list均匀分成num_processes个子集
    emotion_groups = np.array_split(emotion_list, num_processes)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(generate_audio_files_for_emotion_group, url_composer, text_list, group, output_dir_path)
            for group in emotion_groups]
        for future in futures:
            future.result()  # 等待所有进程完成


def generate_audio_files_for_emotion_group(url_composer, text_list, emotion_list, output_dir_path):
    start_time = time.perf_counter()  # 使用 perf_counter 获取高精度计时起点
    # Ensure the output directory exists
    output_dir = os.path.abspath(output_dir_path)
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for text and emotion categories
    text_subdir = os.path.join(output_dir, params.inference_audio_text_aggregation_dir)
    os.makedirs(text_subdir, exist_ok=True)
    emotion_subdir = os.path.join(output_dir, params.inference_audio_emotion_aggregation_dir)
    os.makedirs(emotion_subdir, exist_ok=True)

    all_count = len(text_list) * len(emotion_list)
    has_generated_count = 0
    all_text_count = sum(len(item) for item in text_list)

    # 计算笛卡尔积
    cartesian_product = list(itertools.product(text_list, emotion_list))

    for text, emotion in cartesian_product:
        # Generate audio byte stream using the create_audio function

        emotion_name = emotion['emotion']

        text_subdir_text = os.path.join(text_subdir, text)
        os.makedirs(text_subdir_text, exist_ok=True)
        text_subdir_text_file_path = os.path.join(text_subdir_text, emotion_name + '.wav')

        emotion_subdir_emotion = os.path.join(emotion_subdir, emotion_name)
        os.makedirs(emotion_subdir_emotion, exist_ok=True)
        emotion_subdir_emotion_file_path = os.path.join(emotion_subdir_emotion, text + '.wav')

        # 检查是否已经存在对应的音频文件，如果存在则跳过
        if os.path.exists(text_subdir_text_file_path) and os.path.exists(emotion_subdir_emotion_file_path):
            has_generated_count += 1
            logger.info(f"进程ID: {os.getpid()}, 进度: {has_generated_count}/{all_count}")
            continue

        if url_composer.is_emotion():
            real_url = url_composer.build_url_with_emotion(text, emotion['emotion'], False)
        else:
            real_url = url_composer.build_url_with_ref(text, emotion['ref_path'], emotion['ref_text'], False)

        audio_bytes = inference_audio_from_api(real_url)

        # Write audio bytes to the respective files
        with open(text_subdir_text_file_path, 'wb') as f:
            f.write(audio_bytes)
        with open(emotion_subdir_emotion_file_path, 'wb') as f:
            f.write(audio_bytes)

        has_generated_count += 1
        logger.info(f"进程ID: {os.getpid()}, 进度: {has_generated_count}/{all_count}")
    end_time = time.perf_counter()  # 获取计时终点
    elapsed_time = end_time - start_time  # 计算执行耗时
    # 记录日志内容
    log_message = f"进程ID: {os.getpid()}, generate_audio_files_for_emotion_group 执行耗时: {elapsed_time:.6f} 秒；推理数量: {has_generated_count}； 字符总数：{all_text_count}；每秒推理字符数：{all_text_count*len(emotion_list) / elapsed_time:.3f}；"
    p_logger.info(log_message)
    logger.info(log_message)


def inference_audio_from_api(url):
    logger.info(f'inference_audio_from_api url: {url}')
    # 发起GET请求
    response = requests.get(url, stream=True)

    # 检查响应状态码是否正常（例如200表示成功）
    if response.status_code == 200:
        # 返回音频数据的字节流
        return response.content
    else:
        raise Exception(f"Failed to fetch audio from API. Server responded with status code {response.status_code}.message: {response.json()}")


def start_api_set_model(set_model_url_composer, gpt_models, sovits_models):
    url, post_body = set_model_url_composer.build_post_url([gpt_models, sovits_models], True)
    logger.info(f'set_model_url_composer url: {set_model_url_composer}')
    logger.info(f'start_api_set_model url: {url}')
    logger.info(f'start_api_set_model post_body: {post_body}')
    response = requests.post(url, json=post_body)
    if response.status_code == 200:
        result = response.text
        return result
    else:
        return f'请求失败，状态码：{response.status_code}'


def start_api_v2_set_gpt_model(set_model_url_composer, gpt_models):
    url = set_model_url_composer.build_get_url([gpt_models], False)
    logger.info(f'start_api_v2_set_gpt_model url: {url}')
    response = requests.get(url)
    if response.status_code == 200:
        result = response.text
        return result
    else:
        return f'请求失败，状态码：{response.status_code}'


def start_api_v2_set_sovits_model(set_model_url_composer, sovits_models):
    url = set_model_url_composer.build_get_url([sovits_models], False)
    logger.info(f'start_api_v2_set_sovits_model url: {url}')
    response = requests.get(url)
    if response.status_code == 200:
        result = response.text
        return result
    else:
        return f'请求失败，状态码：{response.status_code}'
