# Hermes Cosmos - Dockerfile
# 基于 Python 3.11-slim 构建，适配 src/hermes/ 项目结构

FROM python:3.11-slim

WORKDIR /app

# Make src/ importable and force unbuffered logs
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件（利用 Docker 层缓存）
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY pyproject.toml .

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8080 9090 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health || exit 1

# 默认命令
CMD ["python", "-m", "hermes.gateway.main"]