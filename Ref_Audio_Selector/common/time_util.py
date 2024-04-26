import time
import os
from Ref_Audio_Selector.config_param.log_config import p_logger
import Ref_Audio_Selector.config_param.config_params as params


def timeit_decorator(func):
    """
    装饰器，用于计算被装饰函数的执行时间。

    参数:
        func (function): 要计时的函数。

    返回:
        function: 包含计时功能的新函数。
    """

    def wrapper(*args, **kwargs):
        if params.time_log_print_type != 'file':
            return func(*args, **kwargs)

        start_time = time.perf_counter()  # 使用 perf_counter 获取高精度计时起点

        func_result = func(*args, **kwargs)  # 执行原函数

        end_time = time.perf_counter()  # 获取计时终点
        elapsed_time = end_time - start_time  # 计算执行耗时

        # 记录日志内容
        log_message = f"进程ID: {os.getpid()}, {func.__name__} 执行耗时: {elapsed_time:.6f} 秒"
        p_logger.info(log_message)

        return func_result

    return wrapper


def time_monitor(func):
    """
        返回结果，追加时间
    """

    def wrapper(*args, **kwargs):

        start_time = time.perf_counter()  # 使用 perf_counter 获取高精度计时起点

        func_result = func(*args, **kwargs)  # 执行原函数

        end_time = time.perf_counter()  # 获取计时终点
        elapsed_time = end_time - start_time  # 计算执行耗时

        return elapsed_time, func_result

    return wrapper


# 使用装饰器
@timeit_decorator
def example_function(n):
    time.sleep(n)  # 假设这是需要计时的函数，这里模拟耗时操作
    return n * 2


def example_function2(n):
    time.sleep(n)  # 假设这是需要计时的函数，这里模拟耗时操作
    return n * 2


if __name__ == "__main__":
    # 调用经过装饰的函数
    # result = example_function(2)
    print(time_monitor(example_function2)(2))
