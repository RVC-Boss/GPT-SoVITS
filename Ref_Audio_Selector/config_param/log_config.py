import logging
import datetime
import Ref_Audio_Selector.config_param.config_params as params


def setup_logging():
    # 获取当前日期，用于文件名和日志内容
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # 创建一个用于常规日志的处理器
    general_handler = logging.FileHandler('general.log', mode='a', encoding='utf-8')
    general_handler.setLevel(logging.INFO)
    general_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    general_handler.setFormatter(general_formatter)

    # 创建一个专用于性能监控日志的处理器
    performance_handler = logging.FileHandler(
        f"{params.time_log_print_dir}/{current_date}.log", mode='a', encoding='utf-8')
    performance_handler.setLevel(logging.INFO)
    performance_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    performance_handler.setFormatter(performance_formatter)

    # 配置一个常规的logger
    general_logger = logging.getLogger('general')
    general_logger.setLevel(logging.INFO)
    general_logger.addHandler(general_handler)

    # 配置一个专门用于性能监控的logger
    performance_logger = logging.getLogger('performance')
    performance_logger.setLevel(logging.INFO)
    performance_logger.addHandler(performance_handler)

    # 配置根logger，以防万一
    logging.basicConfig(level=logging.WARNING, handlers=[general_handler])

    return general_logger, performance_logger


logger, p_logger = setup_logging()