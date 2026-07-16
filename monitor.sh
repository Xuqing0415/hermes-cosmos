#!/bin/bash
# Hermes 
# /

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[MONITOR]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

# 
API_URL="${HERMES_API_URL:-http://localhost:50051}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"  # 5
ALERT_THRESHOLD="${ALERT_THRESHOLD:-0.85}"  # 85%

# 
send_alert() {
    local level=$1
    local message=$2
    echo "[$(date)] [$level] $message" >> /var/log/hermes/monitor.log
    
    # Slack/
    # curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"$message\"}" "$SLACK_WEBHOOK"
}

# 
check_scheduler() {
    log_info "..."
    
    response=$(curl -s "$API_URL/status")
    
    if [ -z "$response" ]; then
        log_err ""
        send_alert "CRITICAL" ""
        return 1
    fi
    
    status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    
    if [ "$status" != "running" ]; then
        log_err ": $status"
        send_alert "ERROR" ": $status"
        return 1
    fi
    
    pending=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['pending_jobs'])")
    running=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['running_jobs'])")
    
    log_ok ""
    log_info "  - Pending jobs: $pending"
    log_info "  - Running jobs: $running"
    
    return 0
}

# 
check_cluster() {
    log_info "..."
    
    response=$(curl -s "$API_URL/cluster/summary")
    
    if [ -z "$response" ]; then
        log_err ""
        return 1
    fi
    
    total=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_gpus'])")
    available=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['available_gpus'])")
    
    utilization=$(( (total - available) * 100 / total ))
    
    log_info "  - GPU: $total"
    log_info "  - GPU: $available"
    log_info "  - : ${utilization}%"
    
    if [ $utilization -gt 95 ]; then
        log_warn "GPU (${utilization}%)"
        send_alert "WARNING" "GPU: ${utilization}%"
    fi
    
    if [ $available -lt 10 ]; then
        log_warn "GPU ( $available )"
        send_alert "WARNING" "GPU: $available"
    fi
    
    return 0
}

# 
check_predictions() {
    log_info "..."
    
    # Agent
    log_info "  - "
    
    return 0
}

# 
generate_daily_report() {
    log_info "..."
    
    REPORT_DIR="/var/log/hermes/reports"
    mkdir -p "$REPORT_DIR"
    
    REPORT_FILE="$REPORT_DIR/daily_report_$(date +%Y%m%d).md"
    
    cat > "$REPORT_FILE" << EOF
# Hermes 

****: $(date +"%Y%m%d")

## 

\`\`\`
$(curl -s "$API_URL/status" | python3 -m json.tool)
\`\`\`

## 

\`\`\`
$(curl -s "$API_URL/cluster/summary" | python3 -m json.tool)
\`\`\`

## 

### 
- : 0
- : 0
- : 0

### 
- : 0

## 

### 
- 

### 
- 

## 

- 
EOF
    
    log_ok ": $REPORT_FILE"
}

# 
main() {
    log_info " Hermes ..."
    
    # 
    mkdir -p /var/log/hermes
    
    while true; do
        echo ""
        log_info "=== : $(date) ==="
        
        check_scheduler
        check_cluster
        check_predictions
        
        # 8
        HOUR=$(date +%H)
        if [ "$HOUR" = "08" ]; then
            generate_daily_report
            # 
            sleep 3600
        fi
        
        # 
        DAY=$(date +%u)
        if [ "$DAY" = "1" ] && [ "$HOUR" = "09" ]; then
            log_info "..."
            # 
            sleep 3600
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# 
case "${1:-monitor}" in
    monitor)
        main
        ;;
    report)
        generate_daily_report
        ;;
    check)
        check_scheduler
        check_cluster
        check_predictions
        ;;
    help|*)
        echo ": $0 {monitor|report|check|help}"
        echo
        echo ":"
        echo "  monitor - "
        echo "  report  - "
        echo "  check   - "
        echo "  help    - "
        ;;
esac
