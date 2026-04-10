import os
import boto3
from typing import Optional
from .file_storage import FileStorageInterface

class ObjectStorage(FileStorageInterface):
    def __init__(self, bucket_name, access_key, secret_key, region='us-east-1'):
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
    
    def store_file(self, file_path: str, content: bytes) -> str:
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file_path)[1]
        key = f"files/{file_id}{file_extension}"
        
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=content
        )
        
        return file_id
    
    def read_file(self, file_id: str) -> bytes:
        # 查找对应的文件
        # 注意：这里简化实现，实际应用中可能需要维护文件ID到S3键的映射
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=f"files/{file_id}"
        )
        
        if 'Contents' in response:
            key = response['Contents'][0]['Key']
            obj = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return obj['Body'].read()
        raise FileNotFoundError(f"File with id {file_id} not found")
    
    def delete_file(self, file_id: str) -> bool:
        # 查找并删除对应的文件
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=f"files/{file_id}"
        )
        
        if 'Contents' in response:
            key = response['Contents'][0]['Key']
            self.s3.delete_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return True
        return False