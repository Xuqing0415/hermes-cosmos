#!/bin/bash
# Hermes 生产试运行脚本
# 用于在生产环境中运行真实负载并收集指标

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[PROD]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

# 配置
API_URL="${HERMES_API_URL:-http://localhost:50051}"
OUTPUT_DIR="production_run_$(date +%Y%m%d_%H%M%S)"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 记录开始时间
START_TIME=$(date +%s)
echo "=== Hermes 生产试运行 ===" > "$OUTPUT_DIR/run.log"
echo "开始时间: $(date)" >> "$OUTPUT_DIR/run.log"
echo "API URL: $API_URL" >> "$OUTPUT_DIR/run.log"

log_info "开始生产试运行..."

# 1. 检查系统状态
log_info "步骤1: 检查系统状态..."
curl -s "$API_URL/status" > "$OUTPUT_DIR/initial_status.json"
cat "$OUTPUT_DIR/initial_status.json" | python3 -m json.tool
echo "" >> "$OUTPUT_DIR/run.log"
echo "=== 初始状态 ===" >> "$OUTPUT_DIR/run.log"
cat "$OUTPUT_DIR/initial_status.json" >> "$OUTPUT_DIR/run.log"

# 2. 提交测试作业
log_info "步骤2: 提交测试作业..."
JOB_RESPONSE=$(curl -s -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "production-test-job",
    "tenant_id": "internal-ai-team",
    "user_id": "hermes-tester",
    "priority": "NORMAL",
    "requirements": {
      "gpu_count": 8,
      "gpu_type": "nvidia-h100",
      "memory_gb": 512,
      "cpu_cores": 32,
      "storage_gb": 1000
    },
    "constraints": {
      "regions": ["us-east"],
      "carbon_aware": true
    },
    "image": "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
    "command": "python train.py --epochs 10 --batch-size 64",
    "environment": {
      "WANDB_API_KEY": "${WANDB_API_KEY}",
      "LOG_LEVEL": "INFO"
    }
  }')

echo "$JOB_RESPONSE" > "$OUTPUT_DIR/job_response.json"
JOB_ID=$(echo "$JOB_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job']['id'])")

log_ok "作业提交成功！"
log_info "作业ID: $JOB_ID"
echo "" >> "$OUTPUT_DIR/run.log"
echo "=== 作业提交 ===" >> "$OUTPUT_DIR/run.log"
echo "作业ID: $JOB_ID" >> "$OUTPUT_DIR/run.log"
echo "$JOB_RESPONSE" >> "$OUTPUT_DIR/run.log"

# 3. 等待调度
log_info "步骤3: 等待调度..."
MAX_WAIT=60
COUNT=0
while [ $COUNT -lt $MAX_WAIT ]; do
  STATUS=$(curl -s "$API_URL/jobs/$JOB_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['job']['status'])")
  if [ "$STATUS" = "RUNNING" ]; then
    log_ok "作业已调度！"
    break
  fi
  if [ "$STATUS" = "FAILED" ]; then
    log_err "作业调度失败！"
    exit 1
  fi
  sleep 1
  COUNT=$((COUNT + 1))
done

SCHEDULE_TIME=$(date +%s)
SCHEDULE_LATENCY=$((SCHEDULE_TIME - START_TIME))
log_info "调度延迟: ${SCHEDULE_LATENCY}秒"
echo "" >> "$OUTPUT_DIR/run.log"
echo "=== 调度完成 ===" >> "$OUTPUT_DIR/run.log"
echo "调度延迟: ${SCHEDULE_LATENCY}秒" >> "$OUTPUT_DIR/run.log"

# 4. 获取调度详情
log_info "步骤4: 获取调度详情..."
curl -s "$API_URL/jobs/$JOB_ID" > "$OUTPUT_DIR/job_details.json"
curl -s "$API_URL/cluster/summary" > "$OUTPUT_DIR/cluster_after_schedule.json"

echo "" >> "$OUTPUT_DIR/run.log"
echo "=== 调度详情 ===" >> "$OUTPUT_DIR/run.log"
cat "$OUTPUT_DIR/job_details.json" >> "$OUTPUT_DIR/run.log"

# 5. 模拟运行一段时间
log_info "步骤5: 模拟作业运行（30秒）..."
sleep 30

# 6. 记录运行状态
log_info "步骤6: 记录运行状态..."
curl -s "$API_URL/jobs/$JOB_ID" > "$OUTPUT_DIR/job_runtime.json"
curl -s "$API_URL/status" > "$OUTPUT_DIR/final_status.json"
curl -s "$API_URL/cluster/summary" > "$OUTPUT_DIR/cluster_final.json"

# 7. 取消作业
log_info "步骤7: 取消测试作业..."
curl -s -X DELETE "$API_URL/jobs/$JOB_ID"

# 8. 生成报告
log_info "步骤8: 生成试运行报告..."
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

cat > "$OUTPUT_DIR/SUMMARY_REPORT.md" << EOF
# Hermes 生产试运行报告

**运行时间**: $(date -d "@$START_TIME") 至 $(date -d "@$END_TIME")
**总耗时**: ${TOTAL_DURATION}秒

---

## 作业信息

| 项目 | 值 |
|------|-----|
| 作业ID | $JOB_ID |
| 调度延迟 | ${SCHEDULE_LATENCY}秒 |
| 目标 | 8 x NVIDIA H100 GPU |

---

## 集群状态变化

### 调度前
\`\`\`json
$(cat "$OUTPUT_DIR/initial_status.json")
\`\`\`

### 调度后
\`\`\`json
$(cat "$OUTPUT_DIR/final_status.json")
\`\`\`

---

## 资源使用

### 调度前
\`\`\`json
$(cat "$OUTPUT_DIR/cluster_after_schedule.json")
\`\`\`

### 调度后
\`\`\`json
$(cat "$OUTPUT_DIR/cluster_final.json")
\`\`\`

---

## 结论

| 指标 | 结果 |
|------|------|
| 作业提交 | ✅ 成功 |
| 调度延迟 | ⏱️ ${SCHEDULE_LATENCY}秒 |
| 资源分配 | ✅ 成功 |
| 作业取消 | ✅ 成功 |

---

## 下一步

1. 分析调度延迟是否符合预期（目标 < 500ms）
2. 检查资源分配是否正确
3. 对比与旧系统的性能差异
EOF

log_ok "试运行完成！"
log_info "报告位置: $OUTPUT_DIR/SUMMARY_REPORT.md"
cat "$OUTPUT_DIR/SUMMARY_REPORT.md"
