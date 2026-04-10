import pymysql
from typing import Dict, Any, List
from .metadata_storage import MetadataStorageInterface

class MySQLStorage(MetadataStorageInterface):
    def __init__(self, host='localhost', port=3306, user='root', password='password', database='vector_db'):
        self.connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            cursorclass=pymysql.cursors.DictCursor
        )
        self._create_table()
    
    def _create_table(self):
        with self.connection.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    id VARCHAR(36) PRIMARY KEY,
                    file_path VARCHAR(255),
                    file_type VARCHAR(50),
                    file_size BIGINT,
                    file_name VARCHAR(255),
                    content_length INT,
                    word_count INT,
                    width INT,
                    height INT,
                    mode VARCHAR(20),
                    fps FLOAT,
                    total_frames INT,
                    duration FLOAT,
                    sample_rate INT,
                    num_samples INT,
                    collection_id VARCHAR(36),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hash VARCHAR(64),
                    extra JSON
                )
            ''')
            self.connection.commit()
    
    def store_metadata(self, metadata: Dict[str, Any]) -> str:
        import uuid
        metadata_id = str(uuid.uuid4())
        metadata['id'] = metadata_id
        
        # 分离extra字段
        extra = {k: v for k, v in metadata.items() if k not in [
            'id', 'file_path', 'file_type', 'file_size', 'file_name',
            'content_length', 'word_count', 'width', 'height', 'mode',
            'fps', 'total_frames', 'duration', 'sample_rate', 'num_samples',
            'collection_id', 'timestamp', 'hash'
        ]}
        metadata['extra'] = extra
        
        with self.connection.cursor() as cursor:
            fields = list(metadata.keys())
            placeholders = ', '.join(['%s'] * len(fields))
            field_names = ', '.join(fields)
            
            sql = f"INSERT INTO metadata ({field_names}) VALUES ({placeholders})"
            cursor.execute(sql, list(metadata.values()))
            self.connection.commit()
        
        return metadata_id
    
    def get_metadata(self, metadata_id: str) -> Dict[str, Any]:
        with self.connection.cursor() as cursor:
            sql = "SELECT * FROM metadata WHERE id = %s"
            cursor.execute(sql, (metadata_id,))
            result = cursor.fetchone()
        return result
    
    def update_metadata(self, metadata_id: str, metadata: Dict[str, Any]) -> bool:
        with self.connection.cursor() as cursor:
            set_clause = ', '.join([f"{k} = %s" for k in metadata.keys()])
            sql = f"UPDATE metadata SET {set_clause} WHERE id = %s"
            values = list(metadata.values()) + [metadata_id]
            cursor.execute(sql, values)
            self.connection.commit()
        return cursor.rowcount > 0
    
    def delete_metadata(self, metadata_id: str) -> bool:
        with self.connection.cursor() as cursor:
            sql = "DELETE FROM metadata WHERE id = %s"
            cursor.execute(sql, (metadata_id,))
            self.connection.commit()
        return cursor.rowcount > 0
    
    def search_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self.connection.cursor() as cursor:
            where_clause = ' AND '.join([f"{k} = %s" for k in filters.keys()])
            sql = f"SELECT * FROM metadata WHERE {where_clause}"
            cursor.execute(sql, list(filters.values()))
            results = cursor.fetchall()
        return results