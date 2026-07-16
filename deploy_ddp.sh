#!/bin/bash
# Hermes DDP

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[DDP]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "=============================================="
echo " Hermes DDP"
echo "=============================================="
echo ""

# kubectl
log_info "1. kubectl..."
if ! command -v kubectl &> /dev/null; then
    log_err "kubectl"
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

# Redis
log_info "3. Redis..."
if ! kubectl get svc redis-service &> /dev/null; then
    log_warn "Redis..."
    kubectl create deployment redis --image=redis:7-alpine
    kubectl expose deployment redis --port=6379 --name=redis-service
    kubectl wait --for=condition=ready pod -l app=redis --timeout=60s
fi
log_ok "Redis"

# PVC
log_info "4. ..."
if ! kubectl get pvc shared-pvc &> /dev/null; then
    cat > /tmp/pvc.yaml << EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
EOF
    kubectl apply -f /tmp/pvc.yaml
fi
log_ok ""

# DDP
log_info "5. DDP..."
docker build -t hermes-ddp:latest -f docker/Dockerfile.ddp .

# Kind
kind load docker-image hermes-ddp:latest --name hermes

log_ok "DDP"

# DDP
log_info "6. DDP..."
python -m uvicorn hermes.scheduler.ddp_scheduler:app --host 0.0.0.0 --port 8001 &
DDP_PID=$!
echo "DDPPID: $DDP_PID"

# 
sleep 5

# DDP
log_info "7. DDP..."
JOB_RESP=$(curl -s -X POST http://localhost:8001/ddp/jobs \
  -H "Content-Type: application/json" \
  -d '{"name":"ddp-test-job","tenant_id":"test","user_id":"test","num_replicas":4}')

JOB_ID=$(echo $JOB_RESP | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id', 'N/A'))")

log_ok "DDPID: $JOB_ID"

echo ""
echo "=============================================="
echo " Hermes DDP!"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "  ID: $JOB_ID"
echo "  : 4"
echo "  StatefulSet: ddp-$JOB_ID"
echo ""
echo " :"
echo ""
echo "  # Pod"
echo "  kubectl get pods -l hermes-job=$JOB_ID -w"
echo ""
echo "  # "
echo "  kubectl logs ddp-$JOB_ID-0 -f"
echo ""
echo "  # Pod"
echo "  kubectl delete pod ddp-$JOB_ID-1"
echo ""
echo "  # "
echo "  curl http://localhost:8001/ddp/jobs/$JOB_ID"
echo ""
echo " : < 30"
echo ""
