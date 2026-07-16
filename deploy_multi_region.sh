#!/bin/bash
# Hermes  - Kind

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
echo " Hermes "
echo "=============================================="
echo ""

# kind
log_info "1. Kind..."
if ! command -v kind &> /dev/null; then
    log_err "Kind"
    exit 1
fi
log_ok "Kind"

# us-east
log_info "2.  us-east ..."
if kind get clusters | grep -q "us-east"; then
    log_warn "us-east..."
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
log_ok "us-east"

# eu-west
log_info "3.  eu-west ..."
if kind get clusters | grep -q "eu-west"; then
    log_warn "eu-west..."
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
log_ok "eu-west"

# asia-east
log_info "4.  asia-east ..."
if kind get clusters | grep -q "asia-east"; then
    log_warn "asia-east..."
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
log_ok "asia-east"

# Redis
log_info "5. Redis..."
kubectl config use-context kind-us-east
kubectl create deployment global-redis --image=redis:7-alpine
kubectl expose deployment global-redis --port=6379 --name=global-redis-service

kubectl config use-context kind-eu-west
kubectl create deployment global-redis --image=redis:7-alpine
kubectl expose deployment global-redis --port=6379 --name=global-redis-service

kubectl config use-context kind-asia-east
kubectl create deployment global-redis --image=redis:7-alpine
kubectl expose deployment global-redis --port=6379 --name=global-redis-service

log_ok "Redis"

# 
log_info "6. Region..."
docker build -t hermes-cross-region:latest -f docker/Dockerfile.cross-region .

kind load docker-image hermes-cross-region:latest --name us-east
kind load docker-image hermes-cross-region:latest --name eu-west
kind load docker-image hermes-cross-region:latest --name asia-east

log_ok ""

# kubeconfig
log_info "7. kubeconfig..."
mkdir -p /etc/hermes/kubeconfig

kubectl config use-context kind-us-east
kubectl config view --minify --flatten > /etc/hermes/kubeconfig/us-east

kubectl config use-context kind-eu-west
kubectl config view --minify --flatten > /etc/hermes/kubeconfig/eu-west

kubectl config use-context kind-asia-east
kubectl config view --minify --flatten > /etc/hermes/kubeconfig/asia-east

log_ok "kubeconfig"

echo ""
echo "=============================================="
echo " !"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "  us-east:     kind-us-east"
echo "  eu-west:     kind-eu-west"
echo "  asia-east:   kind-asia-east"
echo ""
echo " :"
echo ""
echo "  1. Region:"
echo "     python -m uvicorn hermes.scheduler.multi_region_scheduler:app --port 8001"
echo ""
echo "  2. Region:"
echo "     curl -X POST http://localhost:8001/multi-region/jobs \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"name\":\"cross-region-job\",\"tenant_id\":\"test\",\"user_id\":\"test\",\"total_replicas\":4,\"regions\":[\"us-east\",\"eu-west\"]}'"
echo ""
echo "  3. :"
echo "     curl http://localhost:8001/multi-region/jobs/<job-id>"
echo ""
echo "  4. Region:"
echo "     curl -X POST http://localhost:8001/multi-region/jobs/<job-id>/region-failure?failed_region=us-east"
echo ""
