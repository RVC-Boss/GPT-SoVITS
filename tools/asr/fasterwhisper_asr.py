import argparse
import os
import traceback

import torch
from faster_whisper import WhisperModel

from tools.asr.config import fw_model_size_list, BaseASR
from tools.asr.funasr_asr import FunASR
from tools.my_utils import ASR_Logger

os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

language_code_list = [
    "af", "am", "ar", "as", "az", 
    "ba", "be", "bg", "bn", "bo", 
    "br", "bs", "ca", "cs", "cy", 
    "da", "de", "el", "en", "es", 
    "et", "eu", "fa", "fi", "fo", 
    "fr", "gl", "gu", "ha", "haw", 
    "he", "hi", "hr", "ht", "hu", 
    "hy", "id", "is", "it", "ja", 
    "jw", "ka", "kk", "km", "kn", 
    "ko", "la", "lb", "ln", "lo", 
    "lt", "lv", "mg", "mi", "mk", 
    "ml", "mn", "mr", "ms", "mt", 
    "my", "ne", "nl", "nn", "no", 
    "oc", "pa", "pl", "ps", "pt", 
    "ro", "ru", "sa", "sd", "si", 
    "sk", "sl", "sn", "so", "sq", 
    "sr", "su", "sv", "sw", "ta", 
    "te", "tg", "th", "tk", "tl", 
    "tr", "tt", "uk", "ur", "uz", 
    "vi", "yi", "yo", "zh", "yue",
    "auto"]

class FasterWhisperASR(BaseASR):

    def __init__(self, model_size, device="cuda", precision="float16"):
        device, precision = [device, precision] if torch.cuda.is_available() else ["cpu", "float32"]
        self.check_local_models()
        self.model = self.load_model(model_size, device, precision)
        assert self.model is not None, ASR_Logger.error('模型不存在')
        self.zh_model = None

    @classmethod
    def check_local_models(self):
        '''
        启动时检查本地是否有 Faster Whisper 模型.
        '''
        self.model_size_list = fw_model_size_list.copy()
        self.model_path_dict = {}
        for i, size in enumerate(self.model_size_list):
            model_name = f"faster-whisper-{size}"
            model_path, flag = super().check_local_model(
                self, 
                model_name = model_name, 
                model_file = 'model.bin', 
                cache_path = os.path.normpath(os.path.expanduser(f"~/.cache/huggingface/hub/")))
            if flag:
                self.model_size_list[i] = f"{size}-{flag}"
                self.model_path_dict[self.model_size_list[i]] = model_path
        return self.model_size_list

    def load_model(self, model_size, device="cuda", precision="float16"):
        if '-local' in model_size or '-cache' in model_size:
            model_path = self.model_path_dict[model_size]
            model_size = model_size[:-6]
            ASR_Logger.info(f"加载模型: 从 {model_path} 加载 faster-whisper-{model_size} 模型.")
            if 'huggingface' in model_path:
                ASR_Logger.warning(f"可将 {model_path} 移动到 tools/asr/models/ 文件夹下并重命名为 faster-whisper-{model_size}.")
        else:
            model_path = model_size
            ASR_Logger.warning(f"下载模型: 从 https://hf-mirror.com/Systran/faster-whisper-{model_size} 下载 faster-whisper-{model_size} 模型.")

        try:
            model = WhisperModel(model_path, device=device, compute_type=precision)
            if model.model.device != 'cpu':
                device_name = torch.cuda.get_device_name(model.model.device)
            else:
                device_name = 'CPU'
            ASR_Logger.info(f"运行设备: {device_name}, 设定精度: {precision}.")
            ASR_Logger.info(f"创建模型: Faster Whisper 完成.\n")
            return model
        except:
            ASR_Logger.info(traceback.format_exc())
            ASR_Logger.error(f"模型加载失败 or 下载失败, 可访问 https://hf-mirror.com/Systran/faster-whisper-{model_size} 自行下载, 并放置于 tools/asr/models/ 文件夹下")
            return 

    def inference(self, file_path, language='auto'):
        try:
            if language == 'auto': 
                language = None

            segments, info = self.model.transcribe(
                audio          = file_path,
                beam_size      = 5,
                vad_filter     = True,
                vad_parameters = dict(min_silence_duration_ms=700),
                language       = language)

            if info.language == "zh":
                ASR_Logger.info("检测为中文文本, 转 FunASR 处理.")
                if self.zh_model is None:
                    self.zh_model = FunASR()
                text, language = self.zh_model.inference(file_path)
            else:
                text = ''.join([segment.text for segment in segments])
            return text, info.language
        except:
            ASR_Logger.error(f"当前文件 {file_path} 转写失败, 可能不是有效的音频文件.")
            ASR_Logger.error(traceback.format_exc())
            return '', ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file_or_folder", type=str, required=True,
                        help="Input audio file path or folder contain audio files.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, 
                        help="Output folder to store transcriptions.")
    parser.add_argument("-s", "--model_size", type=str, default='large-v3', 
                        choices=FasterWhisperASR.check_local_models(),
                        help="Model Size of Faster Whisper")
    parser.add_argument("-l", "--language", type=str, default='auto',
                        choices=language_code_list,
                        help="Language of the audio files.")
    parser.add_argument("-p", "--precision", type=str, default='float16', 
                        choices=['float16','float32'], help="fp16 or fp32")
    cmd = parser.parse_args()
    ASR = FasterWhisperASR(
        model_size = cmd.model_size,
        precision  = cmd.precision,
    )
    ASR.inference_file_or_folder(
        input_file_or_folder = cmd.input_file_or_folder,
        output_folder        = cmd.output_folder,
        language             = cmd.language,
    )

