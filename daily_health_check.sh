#!/bin/bash
# Hermes 
# 3//

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[CHECK]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

HERMES_API_URL="${HERMES_API_URL:-http://localhost:50051}"
LOG_FILE="daily_log.md"

# 
NOW=$(date +"%Y-%m-%d %H:%M:%S")
PERIOD=""

# 
HOUR=$(date +%H)
if [ $HOUR -ge 6 ] && [ $HOUR -lt 12 ]; then
    PERIOD=""
elif [ $HOUR -ge 12 ] && [ $HOUR -lt 18 ]; then
    PERIOD=""
else
    PERIOD=""
fi

echo ""
echo "=============================================="
echo " Hermes  - $PERIOD"
echo "=============================================="
echo ": $NOW"
echo ""

# 
echo "### $PERIOD ($NOW)" >> "$LOG_FILE"

# 1. 
log_info "1. ..."
SCHEDULER_STATUS=$(curl -s "$HERMES_API_URL/health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status', 'unknown'))")

if [ "$SCHEDULER_STATUS" = "healthy" ]; then
    log_ok ""
    echo "-  : " >> "$LOG_FILE"
else
    log_err ": $SCHEDULER_STATUS"
    echo "-  :  ($SCHEDULER_STATUS)" >> "$LOG_FILE"
fi

# 2. 
log_info "2. ..."
CLUSTER=$(curl -s "$HERMES_API_URL/cluster/summary")
TOTAL_GPUS=$(echo "$CLUSTER" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_gpus', 0))")
AVAILABLE_GPUS=$(echo "$CLUSTER" | python3 -c "import sys,json; print(json.load(sys.stdin).get('available_gpus', 0))")

echo "   GPU: $TOTAL_GPUS | : $AVAILABLE_GPUS"
echo "- GPU: $AVAILABLE_GPUS/$TOTAL_GPUS " >> "$LOG_FILE"

# 3. 
log_info "3. ..."
JOBS=$(curl -s "$HERMES_API_URL/jobs?status=RUNNING")
RUNNING_COUNT=$(echo "$JOBS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total', 0))")

echo "   : $RUNNING_COUNT"
echo "- : $RUNNING_COUNT" >> "$LOG_FILE"

# 4. 
log_info "4. ..."
# 
echo "- : " >> "$LOG_FILE"
log_ok ""

# 5. 
log_info "5. ..."
MEM_USED=$(free -h | grep Mem | awk '{print $3}')
MEM_TOTAL=$(free -h | grep Mem | awk '{print $2}')
DISK_USED=$(df -h / | grep / | awk '{print $3}')
DISK_TOTAL=$(df -h / | grep / | awk '{print $2}')

echo "   : $MEM_USED/$MEM_TOTAL"
echo "   : $DISK_USED/$DISK_TOTAL"
echo "- : $MEM_USED/$MEM_TOTAL" >> "$LOG_FILE"
echo "- : $DISK_USED/$DISK_TOTAL" >> "$LOG_FILE"

echo ""
echo "=============================================="
echo " "
echo "=============================================="
echo ": $LOG_FILE"
echo ""
