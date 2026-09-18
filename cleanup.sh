#!/bin/bash
# Hermes 一键清理脚本
# 清理旧的Checkpoint、日志和问题Pod

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[CLEANUP]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "=============================================="
echo "🧹 Hermes 一键清理"
echo "=============================================="
echo ""

# 1. 清理旧的Checkpoint文件
log_info "1. 清理旧的Checkpoint文件..."
CHECKPOINT_DIR="/var/lib/hermes/checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    # 删除7天前的Checkpoint
    find "$CHECKPOINT_DIR" -type f -mtime +7 -delete
    log_ok "已删除7天前的Checkpoint"
else
    log_warn "Checkpoint目录不存在"
fi

# 2. 清理旧日志
log_info "2. 清理旧日志..."
LOG_DIR="/var/log/hermes"
if [ -d "$LOG_DIR" ]; then
    find "$LOG_DIR" -type f -name "*.log" -mtime +7 -delete
    log_ok "已删除7天前的日志"
else
    log_warn "日志目录不存在"
fi

# 3. 清理停止的容器（Docker环境）
log_info "3. 清理停止的容器..."
docker ps -q --filter "status=exited" | xargs -r docker rm 2>/dev/null || log_warn "无停止的容器"
log_ok "已清理停止的容器"

# 4. 清理未使用的镜像（可选，需要确认）
read -p "是否清理未使用的Docker镜像? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker image prune -f
    log_ok "已清理未使用的镜像"
fi

# 5. 重启有问题的服务
log_info "4. 检查并重启异常服务..."

services=("hermes-scheduler" "hermes-gateway" "hermes-checkpoint")
for service in "${services[@]}"; do
    status=$(docker inspect -f '{{.State.Status}}' "$service" 2>/dev/null || echo "not found")
    if [ "$status" != "running" ]; then
        log_warn "$service 未运行，尝试重启..."
        docker start "$service" 2>/dev/null || log_err "重启失败"
    else
        log_ok "$service 运行正常"
    fi
done

# 6. 清理临时文件
log_info "5. 清理临时文件..."
rm -rf /tmp/hermes_*.tmp 2>/dev/null || true
log_ok "已清理临时文件"

echo ""
echo "=============================================="
echo "✅ 清理完成!"
echo "=============================================="
echo ""
echo "📋 清理内容:"
echo ""
echo "  ✅ 7天前的Checkpoint文件"
echo "  ✅ 7天前的日志文件"
echo "  ✅ 停止的Docker容器"
echo "  ✅ 临时文件"
echo "  ✅ 检查并重启异常服务"
echo ""
echo "💡 建议每周运行一次此脚本"
echo ""
