import os
import pymysql
from typing import Optional
from .file_storage import FileStorageInterface

class DatabaseStorage(FileStorageInterface):
    def __init__(self, host='localhost', port=3306, user='root', password='password', database='vector_db'):
        self.connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        self._create_table()
    
    def _create_table(self):
        with self.connection.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id VARCHAR(36) PRIMARY KEY,
                    file_name VARCHAR(255),
                    file_type VARCHAR(50),
                    file_size BIGINT,
                    content LONGBLOB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.connection.commit()
    
    def store_file(self, file_path: str, content: bytes) -> str:
        file_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        file_type = file_path.split('.')[-1].lower()
        file_size = len(content)
        
        with self.connection.cursor() as cursor:
            sql = "INSERT INTO files (id, file_name, file_type, file_size, content) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(sql, (file_id, file_name, file_type, file_size, content))
            self.connection.commit()
        
        return file_id
    
    def read_file(self, file_id: str) -> bytes:
        with self.connection.cursor() as cursor:
            sql = "SELECT content FROM files WHERE id = %s"
            cursor.execute(sql, (file_id,))
            result = cursor.fetchone()
            if result:
                return result[0]
            raise FileNotFoundError(f"File with id {file_id} not found")
    
    def delete_file(self, file_id: str) -> bool:
        with self.connection.cursor() as cursor:
            sql = "DELETE FROM files WHERE id = %s"
            cursor.execute(sql, (file_id,))
            self.connection.commit()
        return cursor.rowcount > 0