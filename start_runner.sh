#!/bin/bash
# Hermes 
# 7x24

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
echo " Hermes "
echo "=============================================="
echo ""

# 
START_TIME=$(date +%Y-%m-%d_%H%M%S)
echo ": $START_TIME"
echo ""

# 
echo "##  Day1 - " >> daily_log.md
echo "" >> daily_log.md
echo "****: $(date +%Y-%m-%d)" >> daily_log.md
echo "****: $(date +%H:%M:%S)" >> daily_log.md
echo "" >> daily_log.md

# 
log_info "..."

# 
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

echo "..."
TRAIN_RESPONSE=$(curl -s -X POST "$HERMES_API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d "$JOB_CONFIG")

echo "$TRAIN_RESPONSE"
TRAIN_JOB_ID=$(echo "$TRAIN_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job']['id'])")
echo "ID: $TRAIN_JOB_ID"

# 
log_info "..."

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

echo "..."
INFER_RESPONSE=$(curl -s -X POST "$HERMES_API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d "$INFERENCE_CONFIG")

echo "$INFER_RESPONSE"
INFER_JOB_ID=$(echo "$INFER_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job']['id'])")
echo "ID: $INFER_JOB_ID"

# 
echo "### " >> daily_log.md
echo "- ID: $TRAIN_JOB_ID" >> daily_log.md
echo "- ID: $INFER_JOB_ID" >> daily_log.md
echo "- : $(date +%Y-%m-%d %H:%M:%S)" >> daily_log.md
echo "" >> daily_log.md

echo ""
log_ok "!"
echo ""
echo "=============================================="
echo " !"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "  : $TRAIN_JOB_ID"
echo "  : $INFER_JOB_ID"
echo ""
echo " :"
echo ""
echo "  # "
echo "  curl $HERMES_API_URL/jobs/$TRAIN_JOB_ID"
echo "  curl $HERMES_API_URL/jobs/$INFER_JOB_ID"
echo ""
echo "  # "
echo "  curl $HERMES_API_URL/cluster/summary"
echo ""
echo "  # Grafana"
echo "  http://localhost:3000"
echo ""

# ID
echo "$TRAIN_JOB_ID" > /tmp/hermes_train_job_id.txt
echo "$INFER_JOB_ID" > /tmp/hermes_infer_job_id.txt
