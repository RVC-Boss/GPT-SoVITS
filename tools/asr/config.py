import os
from datetime import datetime

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tools.my_utils import ASR_Logger

class BaseASR:
    def __init__(self):
        pass

    def check_local_model(self, model_name, model_file, cache_path):
        '''
        启动时检查本地是否有模型文件夹.
        '''
        # 先检查当前项目是否有模型文件夹
        local_path = os.path.normpath('tools/asr/models')
        model_path = ''
        flag = ''
        for root, dirs, files in os.walk(local_path):
            if model_file in files and model_name + os.sep in os.path.join(root, model_file):
                model_path, flag = root, 'local'
        if not model_path:
            # 当前项目没有则检索本地缓存
            for root, dirs, files in os.walk(cache_path):
                if model_file in files and model_name + os.sep in os.path.join(root, model_file):
                    model_path, flag = root, 'cache'
        return model_path, flag
        
    def load_model(self):
        """
        加载模型.
        """
        raise NotImplementedError
    
    def inference(self):
        """
        对单个文件进行推理, 返回文本, 和相应的语言.
        """
        raise NotImplementedError

    def inference_file_or_folder(self, input_file_or_folder, output_folder, language):
        """
        对文件夹/文件进行推理, 并保存结果.
        """
        assert os.path.exists(input_file_or_folder), ASR_Logger.error('输入路径不存在.')
        if os.path.isfile(input_file_or_folder):
            # 若为文件获取其父目录的文件名
            file_path = input_file_or_folder
            input_file_paths = [os.path.abspath(file_path)]
            output_file_name = os.path.basename(os.path.dirname(file_path))
        else:
            input_folder = input_file_or_folder
            input_file_names = os.listdir(input_folder)
            input_file_names.sort()
            input_file_paths = []
            for input_file_name in input_file_names:
                input_file_path = os.path.abspath(os.path.join(input_folder, input_file_name))
                if os.path.isfile(input_file_path):
                    input_file_paths.append(input_file_path) 

            output_file_name = os.path.basename(input_folder)

        result = []

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if language == 'auto':
            language = None #不设置语种由模型自动输出概率最高的语种
        ASR_Logger.info("开始转写")
        with logging_redirect_tqdm([ASR_Logger]):
            for file_path in tqdm(input_file_paths, desc="转写进度 ", dynamic_ncols=True):
                text, output_language = self.inference(file_path, language)
                if text and output_language:
                    result.append(f"{file_path}|{output_file_name}|{output_language.upper()}|{text}")
        if not result:
            ASR_Logger.error("没有转写结果, 放弃保存.")
            return 

        output_file_path = os.path.abspath(f'{output_folder}/{output_file_name}.list')
        if os.path.exists(output_file_path):
            ASR_Logger.info('输出文件路径已存在, 文件名添加时间戳.')
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            file_name, file_extension = os.path.splitext(output_file_path)
            output_file_path = f"{file_name}-{timestamp}{file_extension}"
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(result))
            ASR_Logger.info(f"任务完成->标注文件路径: {output_file_path}\n")
        return output_file_path
    
fw_model_size_list = [
    "tiny",     "tiny.en", 
    "base",     "base.en", 
    "small",    "small.en", 
    "medium",   "medium.en", 
    "large",    "large-v1", 
    "large-v2", "large-v3"]

asr_dict = {
    "达摩 ASR (中文)": {
        'name': 'funasr',
        'lang': ['zh'],
        'size': ['large'],
        'path': 'funasr_asr.py',
    },
    "Faster Whisper (多语种)": {
        'name': 'fasterwhisper',
        'lang': ['auto', 'zh', 'en', 'ja'],
        'size': fw_model_size_list,
        'path': 'fasterwhisper_asr.py'
    }
}