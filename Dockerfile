FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p data/vector_db data/files data/backups models/embedding models/image logs

# 复制环境变量文件
COPY .env.example .env

# 暴露端口
EXPOSE 8001

# 启动服务
CMD ["uvicorn", "vector_db.api.main:app", "--host", "0.0.0.0", "--port", "8001"]