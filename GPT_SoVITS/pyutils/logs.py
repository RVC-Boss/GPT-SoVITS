import logging
from logging.handlers import RotatingFileHandler

# 设置日志记录器
llog = logging.getLogger(__name__)
llog.setLevel(logging.INFO)
llog.propagate = False  # 防止日志事件传递给根记录器

# 创建控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建文件日志处理器
file_handler = RotatingFileHandler('app.log', maxBytes=1024 * 1024 * 10, backupCount=5)
file_handler.setLevel(logging.INFO)

# 设置日志格式，包括文件名和行号
formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
llog.addHandler(console_handler)
#llog.addHandler(file_handler)