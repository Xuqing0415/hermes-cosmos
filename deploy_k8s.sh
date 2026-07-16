#!/bin/bash
# Hermes K8s

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
echo " Hermes K8s"
echo "=============================================="
echo ""

# kubectl
log_info "1. kubectl..."
if ! command -v kubectl &> /dev/null; then
    log_err "kubectlkubectl"
    exit 1
fi
log_ok "kubectl"

# 
log_info "2. K8s..."
if ! kubectl cluster-info &> /dev/null; then
    log_err "K8s"
    exit 1
fi
log_ok "K8s"

# 
log_info "3. ..."
kubectl apply -f deployment/k8s/namespace.yaml
log_ok "hermes"

# RBAC
log_info "4. RBAC..."
kubectl apply -f deployment/k8s/scheduler-rbac.yaml
log_ok "RBAC"

# Redis
log_info "5. Redis..."
kubectl apply -f deployment/k8s/redis.yaml
log_ok "Redis"

# Redis
log_info "6. Redis..."
kubectl wait --for=condition=ready pod/redis-0 -n hermes --timeout=120s
log_ok "Redis"

# 
log_info "7. ..."
docker build -t hermes-scheduler:latest -f docker/Dockerfile.scheduler .
log_ok ""

# 
log_info "8. ..."
kubectl apply -f deployment/k8s/scheduler-deployment.yaml
log_ok ""

# 
log_info "9. ..."
kubectl wait --for=condition=ready pod -l app=hermes-scheduler -n hermes --timeout=120s
log_ok ""

echo ""
echo "=============================================="
echo " Hermes K8s!"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "  : hermes"
echo "  Redis: 1"
echo "  : 3"
echo ""
echo " :"
echo ""
echo "  # Pod"
echo "  kubectl get pods -n hermes"
echo ""
echo "  # "
echo "  kubectl get svc -n hermes"
echo ""
echo "  # "
echo "  kubectl logs -l app=hermes-scheduler -n hermes -f"
echo ""
echo " :"
echo ""
echo "  docker-compose -f docker-compose-monitor.yml up -d"
echo ""
echo "  # Grafana: http://localhost:3000 (admin/hermes)"
echo "  # Prometheus: http://localhost:9090"
echo ""
