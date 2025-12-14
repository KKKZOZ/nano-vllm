# logger.py
import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def init_logger(
    name: str = "my_llm", level: Optional[int] = None, use_color: bool = True
) -> logging.Logger:
    """
    初始化日志器

    Args:
        name: logger 名称
        level: 日志级别，None 则从环境变量 LOG_LEVEL 读取
        use_color: 是否使用彩色输出
    """
    import os

    # 从环境变量获取日志级别
    if level is None:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # vLLM 风格的格式
    if use_color and sys.stdout.isatty():  # 只在终端中使用颜色
        formatter = ColoredFormatter(
            "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# 全局 logger
logger = init_logger()
