#!/bin/bash
# Hermes 
# 

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

# 
API_URL="${HERMES_API_URL:-http://localhost:50051}"
OUTPUT_DIR="production_run_$(date +%Y%m%d_%H%M%S)"

# 
mkdir -p "$OUTPUT_DIR"

# 
START_TIME=$(date +%s)
echo "=== Hermes  ===" > "$OUTPUT_DIR/run.log"
echo ": $(date)" >> "$OUTPUT_DIR/run.log"
echo "API URL: $API_URL" >> "$OUTPUT_DIR/run.log"

log_info "..."

# 1. 
log_info "1: ..."
curl -s "$API_URL/status" > "$OUTPUT_DIR/initial_status.json"
cat "$OUTPUT_DIR/initial_status.json" | python3 -m json.tool
echo "" >> "$OUTPUT_DIR/run.log"
echo "===  ===" >> "$OUTPUT_DIR/run.log"
cat "$OUTPUT_DIR/initial_status.json" >> "$OUTPUT_DIR/run.log"

# 2. 
log_info "2: ..."
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

log_ok ""
log_info "ID: $JOB_ID"
echo "" >> "$OUTPUT_DIR/run.log"
echo "===  ===" >> "$OUTPUT_DIR/run.log"
echo "ID: $JOB_ID" >> "$OUTPUT_DIR/run.log"
echo "$JOB_RESPONSE" >> "$OUTPUT_DIR/run.log"

# 3. 
log_info "3: ..."
MAX_WAIT=60
COUNT=0
while [ $COUNT -lt $MAX_WAIT ]; do
  STATUS=$(curl -s "$API_URL/jobs/$JOB_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['job']['status'])")
  if [ "$STATUS" = "RUNNING" ]; then
    log_ok ""
    break
  fi
  if [ "$STATUS" = "FAILED" ]; then
    log_err ""
    exit 1
  fi
  sleep 1
  COUNT=$((COUNT + 1))
done

SCHEDULE_TIME=$(date +%s)
SCHEDULE_LATENCY=$((SCHEDULE_TIME - START_TIME))
log_info ": ${SCHEDULE_LATENCY}"
echo "" >> "$OUTPUT_DIR/run.log"
echo "===  ===" >> "$OUTPUT_DIR/run.log"
echo ": ${SCHEDULE_LATENCY}" >> "$OUTPUT_DIR/run.log"

# 4. 
log_info "4: ..."
curl -s "$API_URL/jobs/$JOB_ID" > "$OUTPUT_DIR/job_details.json"
curl -s "$API_URL/cluster/summary" > "$OUTPUT_DIR/cluster_after_schedule.json"

echo "" >> "$OUTPUT_DIR/run.log"
echo "===  ===" >> "$OUTPUT_DIR/run.log"
cat "$OUTPUT_DIR/job_details.json" >> "$OUTPUT_DIR/run.log"

# 5. 
log_info "5: 30..."
sleep 30

# 6. 
log_info "6: ..."
curl -s "$API_URL/jobs/$JOB_ID" > "$OUTPUT_DIR/job_runtime.json"
curl -s "$API_URL/status" > "$OUTPUT_DIR/final_status.json"
curl -s "$API_URL/cluster/summary" > "$OUTPUT_DIR/cluster_final.json"

# 7. 
log_info "7: ..."
curl -s -X DELETE "$API_URL/jobs/$JOB_ID"

# 8. 
log_info "8: ..."
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

cat > "$OUTPUT_DIR/SUMMARY_REPORT.md" << EOF
# Hermes 

****: $(date -d "@$START_TIME")  $(date -d "@$END_TIME")
****: ${TOTAL_DURATION}

---

## 

|  |  |
|------|-----|
| ID | $JOB_ID |
|  | ${SCHEDULE_LATENCY} |
|  | 8 x NVIDIA H100 GPU |

---

## 

### 
\`\`\`json
$(cat "$OUTPUT_DIR/initial_status.json")
\`\`\`

### 
\`\`\`json
$(cat "$OUTPUT_DIR/final_status.json")
\`\`\`

---

## 

### 
\`\`\`json
$(cat "$OUTPUT_DIR/cluster_after_schedule.json")
\`\`\`

### 
\`\`\`json
$(cat "$OUTPUT_DIR/cluster_final.json")
\`\`\`

---

## 

|  |  |
|------|------|
|  |   |
|  | ⏱ ${SCHEDULE_LATENCY} |
|  |   |
|  |   |

---

## 

1.  < 500ms
2. 
3. 
EOF

log_ok ""
log_info ": $OUTPUT_DIR/SUMMARY_REPORT.md"
cat "$OUTPUT_DIR/SUMMARY_REPORT.md"
