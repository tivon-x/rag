import logging
from datetime import datetime
from pathlib import Path

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_filename: str = None,
    console_output: bool = True
) -> logging.Logger:
    """
    设置项目统一的日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志文件存储目录
        log_filename: 日志文件名，如果为None则使用当前日期
        console_output: 是否输出到控制台
    
    Returns:
        配置好的logger实例
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 生成日志文件名
    if log_filename is None:
        log_filename = f"rag_{datetime.now().strftime('%Y%m%d')}.log"
    
    log_file_path = log_path / log_filename
    
    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建根logger
    logger = logging.getLogger('rag')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    获取logger实例
    
    Args:
        name: logger名称，如果为None则返回根logger
    
    Returns:
        logger实例
    """
    if name is None:
        return logging.getLogger('rag')
    return logging.getLogger(f'rag.{name}')

# 初始化默认配置
default_logger = setup_logging()