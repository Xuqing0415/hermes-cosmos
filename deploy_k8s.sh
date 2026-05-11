#!/bin/bash
# Hermes K8s部署脚本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[DEPLOY]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "=============================================="
echo "🚀 Hermes K8s部署"
echo "=============================================="
echo ""

# 检查kubectl
log_info "1. 检查kubectl..."
if ! command -v kubectl &> /dev/null; then
    log_err "kubectl未安装，请先安装kubectl"
    exit 1
fi
log_ok "kubectl已就绪"

# 检查集群连接
log_info "2. 检查K8s集群..."
if ! kubectl cluster-info &> /dev/null; then
    log_err "无法连接K8s集群，请确保集群已启动"
    exit 1
fi
log_ok "K8s集群连接正常"

# 创建命名空间
log_info "3. 创建命名空间..."
kubectl apply -f deployment/k8s/namespace.yaml
log_ok "命名空间hermes创建成功"

# 创建RBAC
log_info "4. 创建RBAC权限..."
kubectl apply -f deployment/k8s/scheduler-rbac.yaml
log_ok "RBAC权限创建成功"

# 部署Redis
log_info "5. 部署Redis..."
kubectl apply -f deployment/k8s/redis.yaml
log_ok "Redis部署成功"

# 等待Redis就绪
log_info "6. 等待Redis就绪..."
kubectl wait --for=condition=ready pod/redis-0 -n hermes --timeout=120s
log_ok "Redis就绪"

# 构建调度器镜像
log_info "7. 构建调度器镜像..."
docker build -t hermes-scheduler:latest -f docker/Dockerfile.scheduler .
log_ok "调度器镜像构建成功"

# 部署调度器
log_info "8. 部署调度器..."
kubectl apply -f deployment/k8s/scheduler-deployment.yaml
log_ok "调度器部署成功"

# 等待调度器就绪
log_info "9. 等待调度器就绪..."
kubectl wait --for=condition=ready pod -l app=hermes-scheduler -n hermes --timeout=120s
log_ok "调度器就绪"

echo ""
echo "=============================================="
echo "✅ Hermes K8s部署完成!"
echo "=============================================="
echo ""
echo "📋 部署清单:"
echo ""
echo "  命名空间: hermes"
echo "  Redis: 1副本"
echo "  调度器: 3副本"
echo ""
echo "💡 验证命令:"
echo ""
echo "  # 查看Pod状态"
echo "  kubectl get pods -n hermes"
echo ""
echo "  # 查看服务"
echo "  kubectl get svc -n hermes"
echo ""
echo "  # 查看日志"
echo "  kubectl logs -l app=hermes-scheduler -n hermes -f"
echo ""
echo "📊 启动监控系统:"
echo ""
echo "  docker-compose -f docker-compose-monitor.yml up -d"
echo ""
echo "  # 访问Grafana: http://localhost:3000 (admin/hermes)"
echo "  # 访问Prometheus: http://localhost:9090"
echo ""
