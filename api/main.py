import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from core.processors.text_processor import TextProcessor
from core.vector_db.faiss_vector_db import FAISSVectorDB
from core.data_flow import data_flow_manager
from core.model_manager import model_manager
from core.cache.cache_manager import cache_manager
from core.security.authentication import auth_manager
from core.backup.backup_manager import backup_manager
from core.logging.log_manager import logger
from config.config import config

# 速率限制器：如果配置了 Redis，使用 Redis 后端实现跨进程共享计数
_cache_backend = config.get('CACHE_BACKEND', 'memory')
if _cache_backend == 'redis':
    _redis_host = config.get('REDIS_HOST', 'localhost')
    _redis_port = config.get('REDIS_PORT', 6379)
    _redis_db = config.get('REDIS_DB', 0)
    _storage_uri = f"redis://{_redis_host}:{_redis_port}/{_redis_db}"
    limiter = Limiter(key_func=get_remote_address, storage_uri=_storage_uri)
else:
    limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 初始化处理器和向量数据库
text_processor = TextProcessor(test_mode=True)
vector_db = FAISSVectorDB()

# 删除已有的测试集合（如果存在）
vector_db.delete_collection("test_collection")
# 创建测试集合
vector_db.create_collection("test_collection", 3)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'.txt', '.md', '.docx', '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mp3', '.wav'}

def _safe_filename(filename: str) -> str:
    """防止路径遍历攻击，只保留文件名部分"""
    return os.path.basename(filename).replace('..', '')

# 认证依赖
def get_current_user(username: Optional[str] = Header(None), api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """获取当前用户"""
    if username and api_key:
        user_info = auth_manager.validate_user(username, api_key)
        if user_info:
            return {"user_id": username, "role": "api", "level": user_info['level']}

    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = auth_manager.validate_jwt(token)
        if payload:
            return payload

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials. Must provide both username and api-key headers.",
        headers={"WWW-Authenticate": "Bearer"},
    )

# 请求和响应模型
class TextProcessRequest(BaseModel):
    text: str

class VectorInsertRequest(BaseModel):
    collection_id: str
    vectors: List[List[float]]
    metadata: List[Dict[str, Any]]

class VectorSearchRequest(BaseModel):
    collection_id: str
    query_vector: List[float]
    top_k: int
    filters: Dict[str, Any] = None

class TextProcessResponse(BaseModel):
    embedding: List[float]
    chunks: List[str]

class VectorInsertResponse(BaseModel):
    vector_ids: List[str]

class VectorSearchResponse(BaseModel):
    results: List[Dict[str, Any]]

class FileUploadResponse(BaseModel):
    file_id: str
    collection_id: str
    chunks_processed: int
    vectors_stored: int

class BatchFileUploadResponse(BaseModel):
    results: List[Dict[str, Any]]

class MetadataRequest(BaseModel):
    file_id: str
    metadata: Dict[str, Any]

class MetadataResponse(BaseModel):
    metadata_id: str
    status: str

class SystemStatusResponse(BaseModel):
    status: str
    config: Dict[str, Any]
    model_status: Dict[str, Any]

class ModelSwitchRequest(BaseModel):
    model_type: str
    model_name: str

class ModelSwitchResponse(BaseModel):
    status: str
    model_path: str

class BackupResponse(BaseModel):
    status: str
    backup_file: str

class BackupListResponse(BaseModel):
    backups: List[str]

class RestoreResponse(BaseModel):
    status: str
    message: str

class APIKeyCreateRequest(BaseModel):
    username: str
    level: int = 0

class APIKeyCreateResponse(BaseModel):
    api_key: str
    username: str
    level: int
    message: str

class APIKeyListResponse(BaseModel):
    api_keys: List[str]

class APIKeyDeleteRequest(BaseModel):
    api_key: str

class APIKeyDeleteResponse(BaseModel):
    status: str
    message: str

class UserListResponse(BaseModel):
    users: List[Dict[str, Any]]


@app.post("/api/auth/apikey/create", response_model=APIKeyCreateResponse)
@limiter.limit("10/minute")
async def create_api_key(request: Request, body: APIKeyCreateRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """生成新的 API Key（需要已认证）"""
    try:
        if current_user.get("role") not in ["admin", "api"]:
            raise HTTPException(status_code=403, detail="Only admin users can create API keys")
        api_key = auth_manager.generate_api_key(body.username, body.level)
        return APIKeyCreateResponse(api_key=api_key, username=body.username, level=body.level, message="API key created successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/apikey/list", response_model=APIKeyListResponse)
@limiter.limit("30/minute")
async def list_api_keys(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    """列出所有 API Key（脱敏显示）"""
    try:
        if current_user.get("role") not in ["admin", "api"]:
            raise HTTPException(status_code=403, detail="Only admin users can list API keys")
        api_keys = auth_manager.list_api_keys()
        return APIKeyListResponse(api_keys=api_keys)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/auth/apikey/delete", response_model=APIKeyDeleteResponse)
@limiter.limit("10/minute")
async def delete_api_key(request: Request, body: APIKeyDeleteRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """删除指定的 API Key"""
    try:
        if current_user.get("role") not in ["admin", "api"]:
            raise HTTPException(status_code=403, detail="Only admin users can delete API keys")
        # 这里需要根据API Key找到对应的用户并删除
        # 暂时使用remove_user方法，需要传入用户名
        # 后续可以添加通过API Key删除用户的功能
        auth_manager.remove_api_key(body.api_key)
        return APIKeyDeleteResponse(status="success", message="API key deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/users/list", response_model=UserListResponse)
@limiter.limit("30/minute")
async def list_users(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    """列出所有用户"""
    try:
        if current_user.get("role") not in ["admin", "api"]:
            raise HTTPException(status_code=403, detail="Only admin users can list users")
        users = auth_manager.list_users()
        return UserListResponse(users=users)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process/text", response_model=TextProcessResponse)
@limiter.limit("60/minute")
async def process_text(request: Request, body: TextProcessRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        chunks = text_processor.chunk(body.text)
        embedding = text_processor.embed(body.text)
        return TextProcessResponse(embedding=embedding, chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vector/insert", response_model=VectorInsertResponse)
@limiter.limit("60/minute")
async def insert_vector(request: Request, body: VectorInsertRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        vector_ids = vector_db.insert(
            body.collection_id,
            body.vectors,
            body.metadata
        )
        return VectorInsertResponse(vector_ids=vector_ids)
    except Exception as e:
        logger.error(f"Insert error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vector/search", response_model=VectorSearchResponse)
@limiter.limit("60/minute")
async def search_vector(request: Request, body: VectorSearchRequest):
    try:
        cache_key = f"search:{body.collection_id}:{hash(tuple(body.query_vector))}:{body.top_k}:{hash(str(body.filters))}"

        cached_results = cache_manager.get(cache_key)
        if cached_results:
            return VectorSearchResponse(results=cached_results)

        results = vector_db.search(
            body.collection_id,
            body.query_vector,
            body.top_k,
            body.filters
        )

        cache_manager.set(cache_key, results)
        return VectorSearchResponse(results=results)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/files/upload", response_model=FileUploadResponse)
@limiter.limit("10/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    collection_id: str = Form(...)
):
    try:
        safe_name = _safe_filename(file.filename)
        ext = os.path.splitext(safe_name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

        import uuid as _uuid
        file_path = f"temp_{_uuid.uuid4().hex}_{safe_name}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        result = data_flow_manager.process_file(file_path, collection_id)

        if os.path.exists(file_path):
            os.remove(file_path)

        return FileUploadResponse(
            file_id=result['file_id'],
            collection_id=result['collection_id'],
            chunks_processed=result['chunks_processed'],
            vectors_stored=result['vectors_stored']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/files/batch/upload", response_model=BatchFileUploadResponse)
@limiter.limit("10/minute")
async def upload_batch_files(
    request: Request,
    files: List[UploadFile] = File(...),
    collection_id: str = Form(...)
):
    try:
        file_paths = []
        for file in files:
            safe_name = _safe_filename(file.filename)
            ext = os.path.splitext(safe_name)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")
            import uuid as _uuid
            file_path = f"temp_{_uuid.uuid4().hex}_{safe_name}"
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            file_paths.append(file_path)

        results = data_flow_manager.process_batch_files(file_paths, collection_id)

        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

        return BatchFileUploadResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metadata/{file_id}")
@limiter.limit("60/minute")
async def get_metadata(request: Request, file_id: str):
    try:
        metadata = data_flow_manager.metadata_storage.get_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Metadata not found")
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metadata/update", response_model=MetadataResponse)
@limiter.limit("60/minute")
async def update_metadata(request: Request, body: MetadataRequest):
    try:
        success = data_flow_manager.metadata_storage.update_metadata(
            body.file_id, body.metadata
        )
        if success:
            return MetadataResponse(metadata_id=body.file_id, status="updated")
        raise HTTPException(status_code=404, detail="Metadata not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/metadata/{file_id}", response_model=MetadataResponse)
@limiter.limit("60/minute")
async def delete_metadata(request: Request, file_id: str):
    try:
        success = data_flow_manager.metadata_storage.delete_metadata(file_id)
        if success:
            return MetadataResponse(metadata_id=file_id, status="deleted")
        raise HTTPException(status_code=404, detail="Metadata not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status", response_model=SystemStatusResponse)
@limiter.limit("30/minute")
async def get_system_status(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        model_status = {
            'text_model': model_manager.check_model_exists(
                config.get('TEXT_PROCESSING_MODEL_NAME'), 'text'
            ),
            'image_model': model_manager.check_model_exists(
                config.get('IMAGE_PROCESSING_MODEL_NAME'), 'image'
            ),
            'embedding_model': model_manager.check_model_exists(
                config.get('EMBEDDING_MODEL_NAME'), 'embedding'
            )
        }
        return SystemStatusResponse(
            status="running",
            config={k: v for k, v in config.get_all().items() if not k.endswith('_PASSWORD')},
            model_status=model_status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/switch", response_model=ModelSwitchResponse)
@limiter.limit("10/minute")
async def switch_model(request: Request, body: ModelSwitchRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        success = model_manager.ensure_model_available(body.model_name, body.model_type)
        if success:
            model_path = model_manager.get_model_path(body.model_name, body.model_type)
            return ModelSwitchResponse(status="switched", model_path=model_path)
        raise HTTPException(status_code=500, detail="Failed to switch model")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backup/create", response_model=BackupResponse)
@limiter.limit("5/minute")
async def create_backup(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        backup_file = backup_manager.create_backup()
        return BackupResponse(status="success", backup_file=backup_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backup/list", response_model=BackupListResponse)
@limiter.limit("30/minute")
async def list_backups(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        backups = backup_manager.list_backups()
        return BackupListResponse(backups=backups)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backup/restore/{backup_file}", response_model=RestoreResponse)
@limiter.limit("5/minute")
async def restore_backup(request: Request, backup_file: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        success = backup_manager.restore_backup(backup_file)
        if success:
            return RestoreResponse(status="success", message="Backup restored successfully")
        raise HTTPException(status_code=500, detail="Failed to restore backup")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/backup/delete/{backup_file}", response_model=RestoreResponse)
@limiter.limit("5/minute")
async def delete_backup(request: Request, backup_file: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        success = backup_manager.delete_backup(backup_file)
        if success:
            return RestoreResponse(status="success", message="Backup deleted successfully")
        raise HTTPException(status_code=500, detail="Failed to delete backup")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
@limiter.limit("120/minute")
async def root(request: Request):
    return {"message": "Vector Database API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.get('API_HOST', '0.0.0.0'), port=config.get('API_PORT', 8000))