#!/bin/bash
# Hermes K8s - Kind

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
echo " Hermes K8s - Kind"
echo "=============================================="
echo ""

# kind
log_info "1. Kind..."
if ! command -v kind &> /dev/null; then
    log_err "KindKind"
    echo ""
    echo " :"
    echo "  Windows:  https://github.com/kubernetes-sigs/kind/releases"
    echo "   kind.exe  PATH"
    exit 1
fi
log_ok "Kind"

# kubectl
log_info "2. kubectl..."
if ! command -v kubectl &> /dev/null; then
    log_err "kubectlkubectl"
    exit 1
fi
log_ok "kubectl"

# Kind
log_info "3. Kind..."
if kind get clusters | grep -q hermes; then
    log_warn " hermes ..."
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

log_ok "Kind"

# kubectl
log_info "4. kubectl..."
kubectl config use-context kind-hermes
log_ok ""

# 
log_info "5. ..."
kubectl wait --for=condition=ready node --all --timeout=120s
log_ok ""

# Redis
log_info "6. Redis..."
kubectl create deployment redis --image=redis:7-alpine
kubectl expose deployment redis --port=6379 --name=redis-service

# Redis
kubectl wait --for=condition=ready pod -l app=redis --timeout=60s
log_ok "Redis"

# kubernetes Python
log_info "7. Python..."
pip install -q kubernetes requests

log_ok "Python"

echo ""
echo "=============================================="
echo " Hermes K8s!"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "  : hermes"
echo "  : 3 (1 + 2)"
echo ""
echo " :"
echo ""
echo "  # "
echo "  kubectl get nodes"
echo ""
echo "  # Pod"
echo "  kubectl get pods"
echo ""
echo "  # Redis"
echo "  kubectl get svc redis-service"
echo ""
echo " :"
echo ""
echo "  1. Hermes:"
echo "     python -m uvicorn hermes.scheduler.main:app --host 0.0.0.0 --port 8001"
echo ""
echo "  2. :"
echo "     curl -X POST http://localhost:8001/jobs \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"name\":\"test-job\",\"tenant_id\":\"test\",\"user_id\":\"test\",\"gpu_count\":1}'"
echo ""
echo "  3. Pod:"
echo "     kubectl get pods -l hermes-job"
echo ""
echo "  4. :"
echo "     kubectl delete pod <pod-name>"
echo ""
echo "  5. :"
echo "     kubectl get pods -w"
echo ""
