import logging
import os
import datetime
import Ref_Audio_Selector.config_param.config_params as params


def create_general_logger():
    # 获取当前日期，用于文件名和日志内容
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # 创建一个用于控制台输出的处理器，并设置日志级别
    console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # 可以设置控制台输出的格式
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.encoding = 'utf-8'  # 设置字符编码为utf-8

    os.makedirs(params.log_dir, exist_ok=True)

    # 创建一个用于常规日志的处理器
    general_handler = logging.FileHandler(f"{params.log_dir}/{current_date}.log", mode='a', encoding='utf-8')
    # general_handler.setLevel(logging.INFO)
    general_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    general_handler.setFormatter(general_formatter)

    # 配置一个常规的logger
    general_logger = logging.getLogger('general')
    level = logging.getLevelName(params.log_level)
    general_logger.setLevel(level)
    general_logger.addHandler(console_handler)
    general_logger.addHandler(general_handler)

    # 配置根logger，以防万一
    logging.basicConfig(level=logging.WARNING, handlers=[general_handler])

    return general_logger


def create_performance_logger():
    # 获取当前日期，用于文件名和日志内容
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    os.makedirs(params.time_log_print_dir, exist_ok=True)

    # 创建一个专用于性能监控日志的处理器
    performance_handler = logging.FileHandler(
        f"{params.time_log_print_dir}/{current_date}.log", mode='a', encoding='utf-8')
    # performance_handler.setLevel(logging.INFO)
    performance_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    performance_handler.setFormatter(performance_formatter)

    # 配置一个专门用于性能监控的logger
    performance_logger = logging.getLogger('performance')
    performance_logger.setLevel(logging.INFO)
    performance_logger.addHandler(performance_handler)

    return performance_logger


def setup_logging():
    return create_general_logger(), create_performance_logger()


logger, p_logger = setup_logging()
