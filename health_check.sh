#!/bin/bash
# Hermes 
# 

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[HEALTH]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

DATE=$(date +%Y-%m-%d)
REPORT_FILE="daily_health_${DATE}.md"
LOG_DIR="health_logs"

mkdir -p "$LOG_DIR"

echo "# Hermes " > "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "****: $DATE" >> "$LOG_DIR/$REPORT_FILE"
echo "****: $(date +%H:%M:%S)" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"

# Docker
echo "##  Docker" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"

if systemctl is-active --quiet docker; then
    echo "|  |  |" >> "$LOG_DIR/$REPORT_FILE"
    echo "|------|------|" >> "$LOG_DIR/$REPORT_FILE"
    echo "| Docker |   |" >> "$LOG_DIR/$REPORT_FILE"
    log_ok "Docker "
else
    echo "| Docker |   |" >> "$LOG_DIR/$REPORT_FILE"
    log_err "Docker "
fi

# Hermes
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "##  Hermes" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "|  |  |" >> "$LOG_DIR/$REPORT_FILE"
echo "|------|------|" >> "$LOG_DIR/$REPORT_FILE"

services=("hermes-redis" "hermes-prometheus" "hermes-grafana" "hermes-jaeger" "hermes-scheduler" "hermes-gateway" "hermes-checkpoint")
failed_count=0

for service in "${services[@]}"; do
    status=$(docker inspect -f '{{.State.Status}}' "$service" 2>/dev/null || echo "unknown")
    if [ "$status" = "running" ]; then
        echo "| $service |   |" >> "$LOG_DIR/$REPORT_FILE"
        log_ok "$service "
    else
        echo "| $service |  $status |" >> "$LOG_DIR/$REPORT_FILE"
        log_err "$service : $status"
        failed_count=$((failed_count + 1))
    fi
done

# API
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "##  API" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "| API |  |  |" >> "$LOG_DIR/$REPORT_FILE"
echo "|-----|------|------|" >> "$LOG_DIR/$REPORT_FILE"

# API
latency=$(curl -s -w '%{time_total}' -o /dev/null http://localhost:50051/health 2>/dev/null || echo "timeout")
if [ "$latency" != "timeout" ]; then
    echo "|  |   | ${latency}s |" >> "$LOG_DIR/$REPORT_FILE"
    log_ok " API "
else
    echo "|  |   | - |" >> "$LOG_DIR/$REPORT_FILE"
    log_err " API "
    failed_count=$((failed_count + 1))
fi

# Gateway API
latency=$(curl -s -w '%{time_total}' -o /dev/null http://localhost:8080/v1/health 2>/dev/null || echo "timeout")
if [ "$latency" != "timeout" ]; then
    echo "| Gateway |   | ${latency}s |" >> "$LOG_DIR/$REPORT_FILE"
    log_ok "Gateway API "
else
    echo "| Gateway |   | - |" >> "$LOG_DIR/$REPORT_FILE"
    log_err "Gateway API "
    failed_count=$((failed_count + 1))
fi

# 
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "##  " >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "|  |  |  |  |  |" >> "$LOG_DIR/$REPORT_FILE"
echo "|--------|------|------|------|--------|" >> "$LOG_DIR/$REPORT_FILE"

df -h --output=target,size,used,avail,pcent / /var/lib/docker 2>/dev/null | grep -v Filesystem | while read -r line; do
    echo "| $line |" >> "$LOG_DIR/$REPORT_FILE"
done

# 
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "##  " >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
mem_total=$(free -h | grep Mem | awk '{print $2}')
mem_used=$(free -h | grep Mem | awk '{print $3}')
mem_available=$(free -h | grep Mem | awk '{print $7}')
mem_usage=$(free | grep Mem | awk '{print $3/$2 * 100.0}' | cut -d. -f1)
echo "|  |  |  |  |" >> "$LOG_DIR/$REPORT_FILE"
echo "|--------|------|------|--------|" >> "$LOG_DIR/$REPORT_FILE"
echo "| $mem_total | $mem_used | $mem_available | ${mem_usage}% |" >> "$LOG_DIR/$REPORT_FILE"

# 
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "##  " >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"

if [ "$failed_count" -eq 0 ]; then
    echo "****:  " >> "$LOG_DIR/$REPORT_FILE"
    echo "****: " >> "$LOG_DIR/$REPORT_FILE"
    log_ok " "
else
    echo "****:   $failed_count " >> "$LOG_DIR/$REPORT_FILE"
    echo "****: " >> "$LOG_DIR/$REPORT_FILE"
    log_err "  $failed_count "
fi

echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "---" >> "$LOG_DIR/$REPORT_FILE"
echo "* $(date +%Y-%m-%d %H:%M:%S)*" >> "$LOG_DIR/$REPORT_FILE"

echo ""
echo "=============================================="
echo " "
echo "=============================================="
echo ": $LOG_DIR/$REPORT_FILE"
echo ": $failed_count"
echo ""

# 
if [ "$failed_count" -gt 0 ]; then
    # 
    echo " "
fi

exit $failed_count
