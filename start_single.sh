#!/bin/bash
# Hermes 单人环境一键启动脚本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[SETUP]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

HERMES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

echo "=============================================="
echo "🚀 Hermes 单人环境一键启动"
echo "=============================================="
echo ""

# 检查依赖
log_info "检查依赖..."

if ! command -v docker &> /dev/null; then
    log_err "Docker 未安装，请先安装 Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_err "Docker Compose 未安装，请先安装"
    exit 1
fi

log_ok "Docker 和 Docker Compose 已就绪"
echo ""

# 构建镜像
log_info "构建 Hermes 镜像..."

cd "$HERMES_DIR"

docker-compose -f docker-compose-single.yml build

log_ok "镜像构建完成"
echo ""

# 启动服务
log_info "启动所有服务..."

docker-compose -f docker-compose-single.yml up -d

log_ok "服务启动中..."
echo ""

# 等待服务启动
log_info "等待服务启动（约30秒）..."
sleep 30

# 检查服务状态
log_info "检查服务状态..."
echo ""

services=("hermes-redis" "hermes-prometheus" "hermes-grafana" "hermes-jaeger" "hermes-scheduler" "hermes-gateway" "hermes-checkpoint")

for service in "${services[@]}"; do
    status=$(docker inspect -f '{{.State.Status}}' "$service" 2>/dev/null || echo "not found")
    if [ "$status" = "running" ]; then
        echo "✅ $service"
    else
        echo "❌ $service ($status)"
    fi
done

echo ""

# 测试调度器
log_info "测试调度器 API..."
if curl -s http://localhost:50051/health &>/dev/null; then
    log_ok "调度器 API 正常"
else
    log_err "调度器 API 不可用"
fi

# 测试 Gateway
log_info "测试 Gateway API..."
if curl -s http://localhost:8080/v1/health &>/dev/null; then
    log_ok "Gateway API 正常"
else
    log_err "Gateway API 不可用"
fi

echo ""
echo "=============================================="
echo "✅ 单人环境启动完成!"
echo "=============================================="
echo ""
echo "📋 服务地址:"
echo ""
echo "  🔧 调度器: http://localhost:50051"
echo "  🌐 Gateway: http://localhost:8080"
echo "  📊 Grafana: http://localhost:3000 (admin/hermes)"
echo "  📈 Prometheus: http://localhost:9090"
echo "  🔍 Jaeger: http://localhost:16686"
echo ""
echo "💡 快速测试命令:"
echo ""
echo "  # 提交测试作业"
echo "  curl -X POST http://localhost:50051/jobs \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"name\":\"test-job\",\"tenant_id\":\"test\",\"user_id\":\"test\",\"requirements\":{\"gpu_count\":1},\"image\":\"pytorch/pytorch:latest\"}'"
echo ""
echo "  # 查看作业状态"
echo "  curl http://localhost:50051/jobs"
echo ""
echo "  # 手动模拟故障（替换为实际Pod名）"
echo "  docker kill hermes-scheduler"
echo "  # 等待5秒后检查是否自动重启"
echo ""
echo "📁 数据目录:"
echo ""
echo "  - Redis: ./data/redis"
echo "  - Prometheus: ./data/prometheus"
echo "  - Grafana: ./data/grafana"
echo "  - Checkpoint: ./data/checkpoints"
echo ""
