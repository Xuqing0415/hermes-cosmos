#!/bin/bash
# Hermes 
# 4

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[WEEK1]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

HERMES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

echo "=============================================="
echo " Hermes "
echo "=============================================="
echo ""

# Python
if ! command -v python3 &> /dev/null; then
    log_err "Python3 "
    exit 1
fi

# 
log_info "..."
pip install pyyaml httpx --quiet 2>/dev/null || true
log_ok ""

# ========== 1:  ==========
log_info "1: ..."

echo ""
echo "1.1  TEE ..."
if grep -q "required: true" "$HERMES_DIR/config/scheduler.yaml" 2>/dev/null; then
    log_ok "TEE "
else
    log_err "TEE "
fi

echo ""
echo "1.2  SIEM ..."
if grep -q "kafka.security.internal" "$HERMES_DIR/config/scheduler.yaml" 2>/dev/null; then
    log_ok "SIEM Kafka "
else
    log_err "SIEM "
fi

echo ""
echo "1.3 Region Checkpoint ..."
if grep -q "replication:" "$HERMES_DIR/config/checkpoint.yaml" 2>/dev/null; then
    log_ok "Region"
else
    log_err "Region"
fi

echo ""
log_ok ""

# ========== 2:  ==========
log_info "2: ..."

if [ -f "$HERMES_DIR/EMAIL_REPORT.md" ]; then
    log_ok ": EMAIL_REPORT.md"
    echo ""
    echo ":"
    echo "1.  EMAIL_REPORT.md"
    echo "2. "
    echo "3. "
else
    log_err ""
fi

# ========== 3:  ==========
log_info "3:  LLM ..."

if [ -f "$HERMES_DIR/config/job_inference.yaml" ]; then
    log_ok ": config/job_inference.yaml"
else
    log_err ""
fi

if [ -f "$HERMES_DIR/inference_test.py" ]; then
    log_ok ": inference_test.py"
else
    log_err ""
fi

# ========== 4:  ==========
log_info "4: ..."

python3 "$HERMES_DIR/daily_standup.py" \
    --completed "" "TEE" "SIEM" "Region" \
    --today "" "" "24" "" \
    --blockers "" \
    --notes ""

# ==========  ==========
echo ""
echo "=============================================="
echo " "
echo "=============================================="
echo ""
echo " :"
echo ""
echo "   PRODUCTION_READINESS_CHECKLIST.md - 50/50"
echo "   config/scheduler.yaml - TEE+SIEM"
echo "   config/checkpoint.yaml - Region"
echo "   config/job_inference.yaml - "
echo "   inference_test.py - 24"
echo "   EMAIL_REPORT.md - "
echo "   DEMO_PRESENTATION.md - "
echo "   daily_standup.py - "
echo ""
echo " :"
echo ""
echo "  : CV"
echo "  : Hermes"
echo "  : "
echo "  -: 24"
echo "  : "
echo ""
echo " :"
echo ""
echo "  # Hermes"
echo "  python -m hermes.scheduler.main &"
echo "  python -m hermes.gateway.main &"
echo ""
echo "  # 24"
echo "  python inference_test.py --config config/job_inference.yaml --duration 24"
echo ""
echo "  # "
echo "  python daily_standup.py --interactive"
echo ""
