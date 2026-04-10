import os
import shutil
import time
import zipfile
from datetime import datetime
from typing import List, Optional
from config.config import config

class BackupManager:
    def __init__(self):
        """初始化备份管理器"""
        self.backup_path = config.get('BACKUP_PATH', './data/backups')
        self.vector_db_path = config.get('VECTOR_DB_PATH', './data/vector_db')
        self.metadata_path = config.get('LOCAL_STORAGE_PATH', './data/files')
        
        # 确保备份目录存在
        os.makedirs(self.backup_path, exist_ok=True)
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """创建备份
        
        Args:
            backup_name: 备份名称，如果不提供则自动生成
            
        Returns:
            备份文件路径
        """
        # 生成备份名称
        if not backup_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f'backup_{timestamp}'
        
        # 创建备份文件
        backup_file = os.path.join(self.backup_path, f'{backup_name}.zip')
        
        # 创建zip文件
        with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 备份向量数据库
            if os.path.exists(self.vector_db_path):
                for root, dirs, files in os.walk(self.vector_db_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.vector_db_path)
                        zipf.write(file_path, arcname=f'vector_db/{arcname}')
            
            # 备份元数据和文件
            if os.path.exists(self.metadata_path):
                for root, dirs, files in os.walk(self.metadata_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.metadata_path)
                        zipf.write(file_path, arcname=f'files/{arcname}')
        
        print(f"备份创建成功: {backup_file}")
        return backup_file
    
    def restore_backup(self, backup_file: str) -> bool:
        """恢复备份
        
        Args:
            backup_file: 备份文件路径
            
        Returns:
            是否恢复成功
        """
        if not os.path.exists(backup_file):
            print(f"备份文件不存在: {backup_file}")
            return False
        
        try:
            # 创建临时目录
            temp_dir = os.path.join(self.backup_path, f'temp_restore_{int(time.time())}')
            os.makedirs(temp_dir, exist_ok=True)
            
            # 解压备份文件
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # 恢复向量数据库
            vector_db_backup = os.path.join(temp_dir, 'vector_db')
            if os.path.exists(vector_db_backup):
                # 备份当前向量数据库
                current_backup = os.path.join(self.backup_path, f'current_vector_db_{int(time.time())}')
                if os.path.exists(self.vector_db_path):
                    shutil.move(self.vector_db_path, current_backup)
                
                # 恢复备份
                shutil.move(vector_db_backup, self.vector_db_path)
                print("向量数据库恢复成功")
            
            # 恢复文件和元数据
            files_backup = os.path.join(temp_dir, 'files')
            if os.path.exists(files_backup):
                # 备份当前文件和元数据
                current_backup = os.path.join(self.backup_path, f'current_files_{int(time.time())}')
                if os.path.exists(self.metadata_path):
                    shutil.move(self.metadata_path, current_backup)
                
                # 恢复备份
                shutil.move(files_backup, self.metadata_path)
                print("文件和元数据恢复成功")
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            print(f"备份恢复成功: {backup_file}")
            return True
        except Exception as e:
            print(f"恢复备份失败: {str(e)}")
            return False
    
    def list_backups(self) -> List[str]:
        """列出所有备份
        
        Returns:
            备份文件列表
        """
        backups = []
        if os.path.exists(self.backup_path):
            for file in os.listdir(self.backup_path):
                if file.endswith('.zip'):
                    backups.append(os.path.join(self.backup_path, file))
        # 按修改时间排序，最新的在前
        backups.sort(key=os.path.getmtime, reverse=True)
        return backups
    
    def delete_backup(self, backup_file: str) -> bool:
        """删除备份
        
        Args:
            backup_file: 备份文件路径
            
        Returns:
            是否删除成功
        """
        if os.path.exists(backup_file):
            try:
                os.remove(backup_file)
                print(f"备份删除成功: {backup_file}")
                return True
            except Exception as e:
                print(f"删除备份失败: {str(e)}")
                return False
        return False
    
    def schedule_backup(self, interval: int = 3600) -> None:
        """定时备份（仅在后台运行）
        
        Args:
            interval: 备份间隔（秒）
        """
        import threading
        
        def backup_task():
            while True:
                self.create_backup()
                time.sleep(interval)
        
        thread = threading.Thread(target=backup_task, daemon=True)
        thread.start()
        print(f"定时备份已启动，间隔: {interval}秒")

# 全局备份管理器实例
backup_manager = BackupManager()