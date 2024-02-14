import logging
import os
import platform
import traceback
import sys
import ffmpeg
import numpy as np


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str):
    if platform.system() == 'Windows':
        path_str = path_str.replace('/', '\\')
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")

COLORS = {
    'WARNING' : '\033[33m',        # Yellow
    'ERROR'   : '\033[31m',        # Red
    'CRITICAL': '\033[35m',
    'RESET'   : '\033[0m',         # Reset color
}

class ColoredConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        # 获取日志级别对应的颜色
        color = COLORS.get(record.levelname, '')
        # 重置颜色
        reset = COLORS['RESET']
        # 设置日志消息的颜色
        self.setFormatter(logging.Formatter(color + '%(asctime)s - %(name)s - %(levelname)s - %(message)s' + reset))
        # 输出日志消息
        super().emit(record)

class Tools_Logger():
    def __init__(self, logger_name, log_level='info', log_path=None):
        assert type(log_level) == str and log_level.upper() in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
        log_level = log_level.upper()
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            ch = ColoredConsoleHandler()
            ch.setLevel(log_level)
            # formatter = logging.Formatter()
            # ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # if log_path is not None:
            #     fh = logging.FileHandler(log_path)
            #     fh.setLevel(log_level)
            #     fh.setFormatter(formatter)
            #     self.logger.addHandler(fh)

    def getLogger(self):
        return self.logger

ASR_Logger = Tools_Logger('ASR').getLogger()