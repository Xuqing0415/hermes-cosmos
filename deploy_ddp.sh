#!/bin/bash
# Hermes DDP分布式训练部署脚本

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
echo "🚀 Hermes DDP分布式训练部署"
echo "=============================================="
echo ""

# 检查kubectl
log_info "1. 检查kubectl..."
if ! command -v kubectl &> /dev/null; then
    log_err "kubectl未安装"
    exit 1
fi
log_ok "kubectl已就绪"

# 检查集群
log_info "2. 检查K8s集群..."
if ! kubectl cluster-info &> /dev/null; then
    log_err "无法连接K8s集群"
    exit 1
fi
log_ok "K8s集群连接正常"

# 检查Redis
log_info "3. 检查Redis服务..."
if ! kubectl get svc redis-service &> /dev/null; then
    log_warn "Redis服务不存在，创建中..."
    kubectl create deployment redis --image=redis:7-alpine
    kubectl expose deployment redis --port=6379 --name=redis-service
    kubectl wait --for=condition=ready pod -l app=redis --timeout=60s
fi
log_ok "Redis服务就绪"

# 创建共享存储PVC
log_info "4. 创建共享存储..."
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
log_ok "共享存储就绪"

# 构建DDP镜像
log_info "5. 构建DDP训练镜像..."
docker build -t hermes-ddp:latest -f docker/Dockerfile.ddp .

# 加载到Kind集群
kind load docker-image hermes-ddp:latest --name hermes

log_ok "DDP镜像构建并加载完成"

# 启动DDP调度器
log_info "6. 启动DDP调度器..."
python -m uvicorn hermes.scheduler.ddp_scheduler:app --host 0.0.0.0 --port 8001 &
DDP_PID=$!
echo "DDP调度器PID: $DDP_PID"

# 等待调度器启动
sleep 5

# 提交DDP作业
log_info "7. 提交DDP训练作业..."
JOB_RESP=$(curl -s -X POST http://localhost:8001/ddp/jobs \
  -H "Content-Type: application/json" \
  -d '{"name":"ddp-test-job","tenant_id":"test","user_id":"test","num_replicas":4}')

JOB_ID=$(echo $JOB_RESP | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id', 'N/A'))")

log_ok "DDP作业提交成功，作业ID: $JOB_ID"

echo ""
echo "=============================================="
echo "✅ Hermes DDP部署完成!"
echo "=============================================="
echo ""
echo "📋 部署信息:"
echo ""
echo "  作业ID: $JOB_ID"
echo "  副本数: 4"
echo "  StatefulSet: ddp-$JOB_ID"
echo ""
echo "💡 验证命令:"
echo ""
echo "  # 查看Pod状态"
echo "  kubectl get pods -l hermes-job=$JOB_ID -w"
echo ""
echo "  # 查看训练日志"
echo "  kubectl logs ddp-$JOB_ID-0 -f"
echo ""
echo "  # 模拟故障（删除一个Pod）"
echo "  kubectl delete pod ddp-$JOB_ID-1"
echo ""
echo "  # 查看作业状态"
echo "  curl http://localhost:8001/ddp/jobs/$JOB_ID"
echo ""
echo "⚡ 预期恢复时间: < 30秒"
echo ""
