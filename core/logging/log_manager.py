import os
import logging
from logging.handlers import RotatingFileHandler
from config.config import config

class LogManager:
    def __init__(self):
        """初始化日志管理器"""
        self.log_file = config.get('LOG_FILE', './logs/app.log')
        self.log_level = getattr(logging, config.get('LOG_LEVEL', 'INFO').upper(), logging.INFO)
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # 配置日志
        self.logger = logging.getLogger('vector_db')
        self.logger.setLevel(self.log_level)
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 创建文件处理器
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.log_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 设置格式化器
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        """获取日志记录器
        
        Returns:
            日志记录器
        """
        return self.logger
    
    def debug(self, message):
        """记录调试信息"""
        self.logger.debug(message)
    
    def info(self, message):
        """记录信息"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误"""
        self.logger.error(message)
    
    def critical(self, message):
        """记录严重错误"""
        self.logger.critical(message)

# 全局日志管理器实例
log_manager = LogManager()
logger = log_manager.get_logger()