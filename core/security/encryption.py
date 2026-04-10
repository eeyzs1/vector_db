import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from config.config import config

class EncryptionManager:
    def __init__(self):
        """初始化加密管理器"""
        # 从配置中获取密钥
        self.key = config.get('ENCRYPTION_KEY', 'default_encryption_key_123456789012345678901234').encode('utf-8')
        # 确保密钥长度为32字节（AES-256）
        if len(self.key) < 32:
            self.key = self.key.ljust(32, b'0')
        elif len(self.key) > 32:
            self.key = self.key[:32]
    
    def encrypt(self, data: bytes) -> bytes:
        """加密数据
        
        Args:
            data: 待加密的数据
            
        Returns:
            加密后的数据
        """
        # 生成随机IV
        iv = os.urandom(16)
        
        # 创建加密器
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # 填充数据
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # 加密
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # 返回IV + 加密数据
        return iv + encrypted_data
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """解密数据
        
        Args:
            encrypted_data: 加密的数据
            
        Returns:
            解密后的数据
        """
        # 提取IV
        iv = encrypted_data[:16]
        data = encrypted_data[16:]
        
        # 创建解密器
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # 解密
        decrypted_padded = decryptor.update(data) + decryptor.finalize()
        
        # 移除填充
        unpadder = padding.PKCS7(128).unpadder()
        decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
        
        return decrypted
    
    def encrypt_string(self, text: str) -> str:
        """加密字符串
        
        Args:
            text: 待加密的字符串
            
        Returns:
            加密后的base64字符串
        """
        encrypted = self.encrypt(text.encode('utf-8'))
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """解密字符串
        
        Args:
            encrypted_text: 加密的base64字符串
            
        Returns:
            解密后的字符串
        """
        encrypted = base64.b64decode(encrypted_text.encode('utf-8'))
        decrypted = self.decrypt(encrypted)
        return decrypted.decode('utf-8')

# 全局加密管理器实例
encryption_manager = EncryptionManager()