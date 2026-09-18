#!/bin/bash
# Hermes K8s部署脚本 - 使用Kind

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
echo "🚀 Hermes K8s部署 - Kind版"
echo "=============================================="
echo ""

# 检查kind
log_info "1. 检查Kind..."
if ! command -v kind &> /dev/null; then
    log_err "Kind未安装，请先安装Kind"
    echo ""
    echo "💡 安装方法:"
    echo "  Windows: 下载 https://github.com/kubernetes-sigs/kind/releases"
    echo "  重命名为 kind.exe 放到 PATH"
    exit 1
fi
log_ok "Kind已就绪"

# 检查kubectl
log_info "2. 检查kubectl..."
if ! command -v kubectl &> /dev/null; then
    log_err "kubectl未安装，请先安装kubectl"
    exit 1
fi
log_ok "kubectl已就绪"

# 创建Kind集群
log_info "3. 创建Kind集群..."
if kind get clusters | grep -q hermes; then
    log_warn "集群 hermes 已存在，删除并重新创建..."
    kind delete cluster --name hermes
fi

cat > /tmp/kind-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
- role: worker
EOF

kind create cluster --name hermes --config /tmp/kind-config.yaml

log_ok "Kind集群创建成功"

# 设置kubectl上下文
log_info "4. 设置kubectl上下文..."
kubectl config use-context kind-hermes
log_ok "上下文切换成功"

# 等待节点就绪
log_info "5. 等待节点就绪..."
kubectl wait --for=condition=ready node --all --timeout=120s
log_ok "节点就绪"

# 部署Redis
log_info "6. 部署Redis服务..."
kubectl create deployment redis --image=redis:7-alpine
kubectl expose deployment redis --port=6379 --name=redis-service

# 等待Redis就绪
kubectl wait --for=condition=ready pod -l app=redis --timeout=60s
log_ok "Redis部署成功"

# 安装kubernetes Python客户端
log_info "7. 安装Python依赖..."
pip install -q kubernetes requests

log_ok "Python依赖安装完成"

echo ""
echo "=============================================="
echo "✅ Hermes K8s环境部署完成!"
echo "=============================================="
echo ""
echo "📋 集群信息:"
echo ""
echo "  集群名称: hermes"
echo "  节点数: 3 (1个控制平面 + 2个工作节点)"
echo ""
echo "💡 验证命令:"
echo ""
echo "  # 查看节点"
echo "  kubectl get nodes"
echo ""
echo "  # 查看Pod"
echo "  kubectl get pods"
echo ""
echo "  # 查看Redis服务"
echo "  kubectl get svc redis-service"
echo ""
echo "🚀 下一步:"
echo ""
echo "  1. 启动Hermes调度器:"
echo "     python -m uvicorn hermes.scheduler.main:app --host 0.0.0.0 --port 8001"
echo ""
echo "  2. 提交测试作业:"
echo "     curl -X POST http://localhost:8001/jobs \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"name\":\"test-job\",\"tenant_id\":\"test\",\"user_id\":\"test\",\"gpu_count\":1}'"
echo ""
echo "  3. 查看Pod:"
echo "     kubectl get pods -l hermes-job"
echo ""
echo "  4. 模拟故障:"
echo "     kubectl delete pod <pod-name>"
echo ""
echo "  5. 观察自动恢复:"
echo "     kubectl get pods -w"
echo ""
