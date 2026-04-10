import os
import uuid
from typing import Optional
from .file_storage import FileStorageInterface

class LocalFileSystemStorage(FileStorageInterface):
    def __init__(self, storage_path='./data/files'):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
    
    def store_file(self, file_path: str, content: bytes) -> str:
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file_path)[1]
        stored_file_path = os.path.join(self.storage_path, f"{file_id}{file_extension}")
        
        with open(stored_file_path, 'wb') as f:
            f.write(content)
        
        return file_id
    
    def read_file(self, file_id: str) -> bytes:
        # 查找对应的文件
        for file_name in os.listdir(self.storage_path):
            if file_name.startswith(file_id):
                file_path = os.path.join(self.storage_path, file_name)
                with open(file_path, 'rb') as f:
                    return f.read()
        raise FileNotFoundError(f"File with id {file_id} not found")
    
    def delete_file(self, file_id: str) -> bool:
        # 查找并删除对应的文件
        for file_name in os.listdir(self.storage_path):
            if file_name.startswith(file_id):
                file_path = os.path.join(self.storage_path, file_name)
                os.remove(file_path)
                return True
        return False