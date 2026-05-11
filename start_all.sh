#!/bin/bash
# Hermes 一键启动脚本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[START]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "=============================================="
echo "🚀 Hermes AI训练调度系统 - 一键启动"
echo "=============================================="
echo ""

# 检查依赖
log_info "1. 检查依赖..."

if ! command -v python &> /dev/null; then
    log_err "Python未安装，请先安装Python 3.10+"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    log_err "pip未安装"
    exit 1
fi

log_ok "Python环境就绪"

# 安装依赖（如果需要）
log_info "2. 安装/检查Python依赖..."
pip install -q streamlit requests fastapi uvicorn httpx

log_ok "依赖安装完成"

# 创建日志目录
mkdir -p logs

echo ""
echo "=============================================="
echo "🎯 启动服务"
echo "=============================================="
echo ""

# 启动API网关（后台运行）
log_info "启动 API Gateway (端口 8000)..."
python -m uvicorn hermes.api_gateway.main:app --host 0.0.0.0 --port 8000 > logs/gateway.log 2>&1 &
GATEWAY_PID=$!
echo "   PID: $GATEWAY_PID"

# 等待启动
sleep 3

# 启动调度器（后台运行）
log_info "启动 Scheduler (端口 8001)..."
python -m uvicorn hermes.scheduler.main:app --host 0.0.0.0 --port 8001 > logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "   PID: $SCHEDULER_PID"

# 等待启动
sleep 3

# 启动Checkpoint服务（后台运行）
log_info "启动 Checkpoint Service (端口 8002)..."
python -m uvicorn hermes.checkpoint.server:app --host 0.0.0.0 --port 8002 > logs/checkpoint.log 2>&1 &
CHECKPOINT_PID=$!
echo "   PID: $CHECKPOINT_PID"

# 等待启动
sleep 3

# 启动Streamlit UI（前台运行）
log_info "启动 Web UI (端口 8501)..."
echo ""
echo "=============================================="
echo "✅ 所有服务启动成功!"
echo "=============================================="
echo ""
echo "📋 服务列表:"
echo ""
echo "  🌐 API Gateway:   http://localhost:8000"
echo "  🚀 Scheduler:     http://localhost:8001"
echo "  💾 Checkpoint:    http://localhost:8002"
echo "  🖥️ Web UI:        http://localhost:8501"
echo ""
echo "💡 访问 Web UI 查看控制面板"
echo ""

# 保存PID到文件
echo "$GATEWAY_PID" > logs/gateway.pid
echo "$SCHEDULER_PID" > logs/scheduler.pid
echo "$CHECKPOINT_PID" > logs/checkpoint.pid

# 启动Streamlit
streamlit run ui/app.py
