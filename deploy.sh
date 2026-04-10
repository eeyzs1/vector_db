#!/bin/bash

# 部署脚本

echo "开始部署向量数据库系统..."

# 检查Python版本
echo "检查Python版本..."
python3 --version

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p data/vector_db
data/files
data/backups
models/embedding
models/image
logs

# 复制环境变量文件
echo "配置环境变量..."
cp .env.example .env

# 启动服务
echo "启动服务..."
uvicorn vector_db.api.main:app --host 0.0.0.0 --port 8001 --reload