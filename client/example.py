"""客户端使用示例"""
from client import VectorDBClient

# ========== 初始化客户端 ==========

# 使用 API Key 认证（新系统必须提供用户名）
client = VectorDBClient(
    base_url="http://localhost:8000",
    username="your-username",  # 必需，用于新的认证系统
    api_key="your-api-key"
)

# 或使用 JWT Token 认证
# client = VectorDBClient(
#     base_url="http://localhost:8000",
#     jwt_token="your-jwt-token"
# )

# ========== API Key 管理 ==========

# 生成新的 API Key（需要管理员权限）
# 注意：新系统需要提供用户名和级别参数
new_key_result = client.create_api_key(username="new_user", level=5)
print(f"新生成的 API Key: {new_key_result['api_key']}")
print(f"用户名: {new_key_result['username']}")
print(f"级别: {new_key_result['level']}")

# 列出所有 API Key（脱敏显示）
keys = client.list_api_keys()
print(f"所有 API Key: {keys}")

# 删除指定 API Key
# client.delete_api_key("vdb_xxx")

# ========== 文本处理 ==========

# 处理文本：分块 + 嵌入
result = client.process_text("这是一段测试文本，用于演示向量数据库的功能。")
print(f"文本嵌入维度: {len(result['embedding'])}")
print(f"分块数量: {len(result['chunks'])}")

# ========== 文件上传 ==========

# 上传单个文件
upload_result = client.upload_file(
    file_path="./document.pdf",
    collection_id="my_collection"
)
print(f"文件 ID: {upload_result['file_id']}")
print(f"处理的分块数: {upload_result['chunks_processed']}")
print(f"存储的向量数: {upload_result['vectors_stored']}")

# 批量上传文件
batch_results = client.batch_upload_files(
    file_paths=["./doc1.pdf", "./doc2.txt", "./image.jpg"],
    collection_id="my_collection"
)
for result in batch_results:
    print(f"文件 {result['file_id']} 处理完成")

# ========== 向量搜索 ==========

# 相似度搜索
query_vector = result['embedding']  # 使用之前生成的嵌入向量
search_results = client.search_vectors(
    collection_id="my_collection",
    query_vector=query_vector,
    top_k=5,
    filters={"file_type": "pdf"}  # 可选的元数据过滤
)

for i, res in enumerate(search_results):
    print(f"结果 {i+1}:")
    print(f"  相似度: {res['score']}")
    print(f"  文件 ID: {res['metadata']['file_id']}")
    print(f"  文件类型: {res['metadata']['file_type']}")

# ========== 元数据操作 ==========

# 获取文件元数据
file_id = upload_result['file_id']
metadata = client.get_metadata(file_id)
print(f"文件元数据: {metadata}")

# 更新元数据
client.update_metadata(
    file_id=file_id,
    metadata={"tags": ["重要", "已审核"], "category": "技术文档"}
)

# ========== 系统管理 ==========

# 获取系统状态
status = client.get_system_status()
print(f"系统状态: {status['status']}")
print(f"当前模型: {status['model_status']}")

# 切换模型
# model_path = client.switch_model(
#     model_type="embedding",
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )
# print(f"模型已切换: {model_path}")

# ========== 备份管理 ==========

# 创建备份
backup_file = client.create_backup()
print(f"备份已创建: {backup_file}")

# 列出所有备份
backups = client.list_backups()
print(f"可用备份: {backups}")

# 恢复备份
# client.restore_backup("backup_20260316.tar.gz")

# 删除备份
# client.delete_backup("old_backup.tar.gz")