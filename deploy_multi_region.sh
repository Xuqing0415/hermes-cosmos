#!/bin/bash
# Hermes 多集群部署脚本 - 创建多个Kind集群

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[MULTI-REGION]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "=============================================="
echo "🌍 Hermes 多集群部署"
echo "=============================================="
echo ""

# 检查kind
log_info "1. 检查Kind..."
if ! command -v kind &> /dev/null; then
    log_err "Kind未安装"
    exit 1
fi
log_ok "Kind已就绪"

# 创建us-east集群
log_info "2. 创建 us-east 集群..."
if kind get clusters | grep -q "us-east"; then
    log_warn "us-east集群已存在，删除重建..."
    kind delete cluster --name us-east
fi

cat > /tmp/us-east-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
networking:
  podSubnet: "10.244.0.0/16"
  serviceSubnet: "10.96.0.0/12"
EOF

kind create cluster --name us-east --config /tmp/us-east-config.yaml
log_ok "us-east集群创建成功"

# 创建eu-west集群
log_info "3. 创建 eu-west 集群..."
if kind get clusters | grep -q "eu-west"; then
    log_warn "eu-west集群已存在，删除重建..."
    kind delete cluster --name eu-west
fi

cat > /tmp/eu-west-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
networking:
  podSubnet: "10.245.0.0/16"
  serviceSubnet: "10.97.0.0/12"
EOF

kind create cluster --name eu-west --config /tmp/eu-west-config.yaml
log_ok "eu-west集群创建成功"

# 创建asia-east集群（可选）
log_info "4. 创建 asia-east 集群..."
if kind get clusters | grep -q "asia-east"; then
    log_warn "asia-east集群已存在，删除重建..."
    kind delete cluster --name asia-east
fi

cat > /tmp/asia-east-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
networking:
  podSubnet: "10.246.0.0/16"
  serviceSubnet: "10.98.0.0/12"
EOF

kind create cluster --name asia-east --config /tmp/asia-east-config.yaml
log_ok "asia-east集群创建成功"

# 部署全局Redis
log_info "5. 部署全局Redis..."
kubectl config use-context kind-us-east
kubectl create deployment global-redis --image=redis:7-alpine
kubectl expose deployment global-redis --port=6379 --name=global-redis-service

kubectl config use-context kind-eu-west
kubectl create deployment global-redis --image=redis:7-alpine
kubectl expose deployment global-redis --port=6379 --name=global-redis-service

kubectl config use-context kind-asia-east
kubectl create deployment global-redis --image=redis:7-alpine
kubectl expose deployment global-redis --port=6379 --name=global-redis-service

log_ok "全局Redis部署完成"

# 构建并加载镜像
log_info "6. 构建跨Region训练镜像..."
docker build -t hermes-cross-region:latest -f docker/Dockerfile.cross-region .

kind load docker-image hermes-cross-region:latest --name us-east
kind load docker-image hermes-cross-region:latest --name eu-west
kind load docker-image hermes-cross-region:latest --name asia-east

log_ok "镜像构建并加载完成"

# 保存kubeconfig
log_info "7. 保存kubeconfig..."
mkdir -p /etc/hermes/kubeconfig

kubectl config use-context kind-us-east
kubectl config view --minify --flatten > /etc/hermes/kubeconfig/us-east

kubectl config use-context kind-eu-west
kubectl config view --minify --flatten > /etc/hermes/kubeconfig/eu-west

kubectl config use-context kind-asia-east
kubectl config view --minify --flatten > /etc/hermes/kubeconfig/asia-east

log_ok "kubeconfig保存完成"

echo ""
echo "=============================================="
echo "✅ 多集群部署完成!"
echo "=============================================="
echo ""
echo "📋 集群列表:"
echo ""
echo "  us-east:     kind-us-east"
echo "  eu-west:     kind-eu-west"
echo "  asia-east:   kind-asia-east"
echo ""
echo "💡 下一步:"
echo ""
echo "  1. 启动多Region调度器:"
echo "     python -m uvicorn hermes.scheduler.multi_region_scheduler:app --port 8001"
echo ""
echo "  2. 提交跨Region作业:"
echo "     curl -X POST http://localhost:8001/multi-region/jobs \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"name\":\"cross-region-job\",\"tenant_id\":\"test\",\"user_id\":\"test\",\"total_replicas\":4,\"regions\":[\"us-east\",\"eu-west\"]}'"
echo ""
echo "  3. 查看作业状态:"
echo "     curl http://localhost:8001/multi-region/jobs/<job-id>"
echo ""
echo "  4. 模拟Region故障:"
echo "     curl -X POST http://localhost:8001/multi-region/jobs/<job-id>/region-failure?failed_region=us-east"
echo ""
