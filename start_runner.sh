#!/bin/bash
# Hermes 陪跑作业启动脚本
# 用于7x24小时持续运行，验证系统稳定性

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[RUNNER]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

HERMES_API_URL="${HERMES_API_URL:-http://localhost:50051}"

echo "=============================================="
echo "🚀 Hermes 陪跑作业启动"
echo "=============================================="
echo ""

# 记录开始时间
START_TIME=$(date +%Y-%m-%d_%H%M%S)
echo "开始时间: $START_TIME"
echo ""

# 更新日志文件
echo "## 📅 Day1 - 部署陪跑作业" >> daily_log.md
echo "" >> daily_log.md
echo "**日期**: $(date +%Y-%m-%d)" >> daily_log.md
echo "**开始时间**: $(date +%H:%M:%S)" >> daily_log.md
echo "" >> daily_log.md

# 启动训练循环作业
log_info "启动训练循环作业..."

# 创建训练作业配置
JOB_CONFIG=$(cat <<EOF
{
  "name": "hermes-runner-training",
  "tenant_id": "runner",
  "user_id": "runner",
  "priority": "NORMAL",
  "requirements": {
    "gpu_count": 1,
    "memory_gb": 64,
    "cpu_cores": 8,
    "storage_gb": 100,
    "max_duration_hours": 168
  },
  "constraints": {},
  "image": "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
  "command": "python -c \"import torch; import time; while True: x = torch.randn(1000, 1000); y = x @ x.T; print('Training step done'); time.sleep(60)\"",
  "checkpoint_enabled": true
}
EOF
)

echo "提交训练作业..."
TRAIN_RESPONSE=$(curl -s -X POST "$HERMES_API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d "$JOB_CONFIG")

echo "$TRAIN_RESPONSE"
TRAIN_JOB_ID=$(echo "$TRAIN_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job']['id'])")
echo "训练作业ID: $TRAIN_JOB_ID"

# 启动推理服务
log_info "启动推理服务..."

INFERENCE_CONFIG=$(cat <<EOF
{
  "name": "hermes-runner-inference",
  "tenant_id": "runner",
  "user_id": "runner",
  "priority": "HIGH",
  "requirements": {
    "gpu_count": 1,
    "memory_gb": 64,
    "cpu_cores": 8,
    "storage_gb": 100,
    "max_duration_hours": 168
  },
  "constraints": {},
  "image": "ghcr.io/huggingface/text-generation-inference:latest",
  "command": "--model-id gpt2 --port 8080",
  "checkpoint_enabled": false
}
EOF
)

echo "提交推理服务..."
INFER_RESPONSE=$(curl -s -X POST "$HERMES_API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d "$INFERENCE_CONFIG")

echo "$INFER_RESPONSE"
INFER_JOB_ID=$(echo "$INFER_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job']['id'])")
echo "推理服务ID: $INFER_JOB_ID"

# 更新日志
echo "### 运行状态" >> daily_log.md
echo "- 训练作业ID: $TRAIN_JOB_ID" >> daily_log.md
echo "- 推理服务ID: $INFER_JOB_ID" >> daily_log.md
echo "- 系统启动时间: $(date +%Y-%m-%d %H:%M:%S)" >> daily_log.md
echo "" >> daily_log.md

echo ""
log_ok "陪跑作业已启动!"
echo ""
echo "=============================================="
echo "✅ 陪跑作业启动完成!"
echo "=============================================="
echo ""
echo "📋 作业信息:"
echo ""
echo "  训练作业: $TRAIN_JOB_ID"
echo "  推理服务: $INFER_JOB_ID"
echo ""
echo "💡 监控命令:"
echo ""
echo "  # 查看作业状态"
echo "  curl $HERMES_API_URL/jobs/$TRAIN_JOB_ID"
echo "  curl $HERMES_API_URL/jobs/$INFER_JOB_ID"
echo ""
echo "  # 查看集群状态"
echo "  curl $HERMES_API_URL/cluster/summary"
echo ""
echo "  # Grafana监控"
echo "  http://localhost:3000"
echo ""

# 保存作业ID到文件
echo "$TRAIN_JOB_ID" > /tmp/hermes_train_job_id.txt
echo "$INFER_JOB_ID" > /tmp/hermes_infer_job_id.txt
