# Vector Database

一个高效的向量相似度搜索引擎和向量数据库系统，支持多种文件类型的处理、向量嵌入、存储和检索。

## 功能特性

- 多文件类型支持：文本（txt/md/docx/pdf）、图像（jpg/png/bmp）、视频（mp4/avi/mov）、音频（mp3/wav）
- 多向量数据库后端：FAISS、HNSWlib、Annoy
- 多元数据存储后端：MySQL、Redis、MongoDB
- 多文件存储后端：本地文件系统、S3、数据库
- RESTful API（FastAPI）
- JWT + API Key 双重认证
- 跨进程共享缓存（Redis）
- 跨进程速率限制（Redis）
- 多进程安全的 FAISS 文件锁
- 自动备份与恢复
- 本地模型自动下载（支持国内镜像）

## 项目结构

```
vector_db/
├── api/
│   └── main.py              # FastAPI 入口，所有 REST 端点
├── config/
│   ├── config.py            # 配置管理（读取 .env）
│   └── .env.example         # 配置模板
├── core/
│   ├── backup/              # 备份与恢复
│   ├── cache/               # LRU 缓存（内存 / Redis）
│   ├── llm/                 # LLM 接口（本地 transformers / 远程 OpenAI 兼容）
│   ├── logging/             # 日志管理
│   ├── model_manager.py     # 模型下载与生命周期管理
│   ├── processors/          # 文本 / 图像 / 视频 / 音频处理器
│   ├── readers/             # 文件读取器
│   ├── security/            # JWT 认证 + AES-256 加密
│   ├── storage/             # 文件存储 + 元数据存储实现
│   ├── utils/               # 并行处理工具
│   ├── vector_db/           # FAISS / HNSW / Annoy 向量数据库实现
│   └── data_flow.py         # 核心数据流程编排
├── data/                    # 运行时数据（向量索引、文件、备份）
├── models/                  # 本地模型缓存
├── tests/
│   ├── unit_tests/          # 单元测试
│   └── load_test.py         # Locust 压力测试
├── requirements.txt
└── main.py                  # 直接运行入口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp config/.env.example .env
```

编辑 `.env`，至少修改以下项：

```env
API_SECRET_KEY=your-secret-key   # 必须修改
API_KEY=your-initial-api-key     # 初始 API Key

# 向量数据库
VECTOR_DB_TYPE=faiss             # faiss | hnsw | annoy
VECTOR_DB_PATH=./data/vector_db

# 元数据存储
METADATA_STORAGE_TYPE=mysql      # mysql | redis | mongodb
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=vector_db

# 缓存（单进程用 memory，多进程用 redis）
CACHE_BACKEND=memory             # memory | redis
```

### 3. 启动服务

```bash
# 开发模式（TEST_MODE 跳过真实模型加载）
TEST_MODE=true uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# 生产模式（单进程）
uvicorn api.main:app --host 0.0.0.0 --port 8001

# 生产模式（多进程，需要 Redis）
CACHE_BACKEND=redis uvicorn api.main:app --host 0.0.0.0 --port 8001 --workers 4
```

## Python 客户端使用

项目提供了 Python SDK，简化 API 调用：

```python
from client import VectorDBClient

# 初始化客户端
client = VectorDBClient(
    base_url="http://localhost:8001",
    api_key="your-api-key"
)

# 上传文件并向量化
result = client.upload_file(
    file_path="./document.pdf",
    collection_id="my_collection"
)
print(f"处理了 {result['chunks_processed']} 个分块")

# 文本搜索
text_result = client.process_text("查询文本")
search_results = client.search_vectors(
    collection_id="my_collection",
    query_vector=text_result['embedding'],
    top_k=5
)

for res in search_results:
    print(f"相似度: {res['score']}, 文件: {res['metadata']['file_id']}")
```

完整示例见 [client/example.py](client/example.py)

## API 文档

启动后访问 `http://localhost:8001/docs` 查看交互式 Swagger 文档。

### 认证

所有需要认证的端点支持两种方式：

```
# API Key
api-key: your-api-key

# JWT Bearer Token
Authorization: Bearer <token>
```

### 主要端点

| 方法 | 路径 | 说明 | 限流 |
|------|------|------|------|
| POST | `/api/auth/apikey/create` | 生成新 API Key（需 admin 权限） | 10/min |
| GET  | `/api/auth/apikey/list` | 列出所有 API Key（脱敏） | 30/min |
| DELETE | `/api/auth/apikey/delete` | 删除指定 API Key | 10/min |
| POST | `/api/process/text` | 文本分块 + 向量嵌入 | 60/min |
| POST | `/api/vector/insert` | 插入向量 | 60/min |
| POST | `/api/vector/search` | 向量相似度搜索 | 60/min |
| POST | `/api/files/upload` | 上传单文件并处理 | 10/min |
| POST | `/api/files/batch/upload` | 批量上传文件 | 10/min |
| GET  | `/api/metadata/{file_id}` | 获取文件元数据 | 60/min |
| POST | `/api/metadata/update` | 更新元数据 | 60/min |
| DELETE | `/api/metadata/{file_id}` | 删除元数据 | 60/min |
| GET  | `/api/system/status` | 系统状态 | 30/min |
| POST | `/api/model/switch` | 切换模型 | 10/min |
| POST | `/api/backup/create` | 创建备份 | 5/min |
| GET  | `/api/backup/list` | 备份列表 | 30/min |
| POST | `/api/backup/restore/{file}` | 恢复备份 | 5/min |
| DELETE | `/api/backup/delete/{file}` | 删除备份 | 5/min |

### 示例：上传文件并搜索

```bash
# 上传文件
curl -X POST http://localhost:8001/api/files/upload \
  -H "api-key: your-api-key" \
  -F "file=@document.pdf" \
  -F "collection_id=my_collection"

# 向量搜索
curl -X POST http://localhost:8001/api/vector/search \
  -H "Content-Type: application/json" \
  -d '{
    "collection_id": "my_collection",
    "query_vector": [0.1, 0.2, ...],
    "top_k": 5,
    "filters": {"file_type": "pdf"}
  }'
```

## 配置说明

### 模型配置

```env
# 文本嵌入（本地 sentence-transformers 或远程 OpenAI 兼容接口）
TEXT_PROCESSING_MODEL_TYPE=local        # local | remote
TEXT_PROCESSING_MODEL_NAME=shibing624/text2vec-base-chinese
TEXT_PROCESSING_API_KEY=                # remote 模式时填写

# 图像嵌入
IMAGE_PROCESSING_MODEL_TYPE=local
IMAGE_PROCESSING_MODEL_NAME=OFA-Sys/chinese-clip-vit-base-patch16
```

本地模型首次运行时自动从 HuggingFace 下载，支持国内镜像（SJTU、hf-mirror、ModelScope）。

### 多进程部署

多进程模式下需要 Redis 来共享缓存和速率限制计数器：

```env
CACHE_BACKEND=redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

FAISS 写操作通过文件锁（`filelock`）保证多进程安全，无需额外配置。

## 运行测试

```bash
# 单元测试
cd vector_db
TEST_MODE=true pytest tests/unit_tests/ -v

# 压力测试（需先启动服务）
locust -f tests/load_test.py --host http://localhost:8001
```

## 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | FastAPI + uvicorn |
| 向量数据库 | FAISS / HNSWlib / Annoy |
| 文本嵌入 | sentence-transformers / transformers |
| 图像/音频嵌入 | CLIP / librosa MFCC |
| 元数据存储 | MySQL / Redis / MongoDB |
| 文件存储 | 本地 / S3 / 数据库 |
| 缓存 | 内存 LRU / Redis |
| 认证 | JWT (HS256) + API Key |
| 加密 | AES-256 CBC |
| 速率限制 | slowapi（支持 Redis 后端） |
| 并发安全 | filelock（FAISS 写锁） |
