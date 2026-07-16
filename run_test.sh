#!/bin/bash
# Hermes 
# 

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[TEST]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

# 
HERMES_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PYTHON="${PYTHON:-python3}"

# 
check_dependencies() {
    log_info "..."
    
    # Python
    if ! command -v "$PYTHON" &> /dev/null; then
        log_err "Python  Python 3.10+"
        exit 1
    fi
    
    # pip
    if ! command -v pip &> /dev/null; then
        log_err "pip "
        exit 1
    fi
    
    # 
    if ! "$PYTHON" -c "import fastapi; import httpx" &> /dev/null; then
        log_info "..."
        pip install -e "$HERMES_DIR"
    fi
    
    log_ok ""
}

# 
start_services() {
    log_info " Hermes ..."
    
    # 
    if curl -s http://localhost:50051/status &> /dev/null; then
        log_warn "Hermes "
        return 0
    fi
    
    # 
    log_info "..."
    "$PYTHON" -m hermes.scheduler.main &
    SCHEDULER_PID=$!
    sleep 3
    
    # Gateway
    log_info " Gateway..."
    "$PYTHON" -m hermes.gateway.main &
    GATEWAY_PID=$!
    sleep 3
    
    # Checkpoint
    log_info " Checkpoint ..."
    "$PYTHON" -m hermes.checkpoint.main &
    CHECKPOINT_PID=$!
    sleep 3
    
    # 
    log_info "..."
    for i in {1..10}; do
        if curl -s http://localhost:50051/status &> /dev/null; then
            log_ok ""
            return 0
        fi
        sleep 2
    done
    
    log_err ""
    return 1
}

# 
run_test_job() {
    log_info "..."
    
    # 
    export HERMES_API_URL=http://localhost:50051
    
    # 
    "$PYTHON" "$HERMES_DIR/examples/submit_job.py" \
        --job-name "hermes-first-run" \
        --gpu-count 8 \
        --max-wait 30
    
    log_ok ""
}

# 
generate_report() {
    log_info "..."
    
    REPORT_FILE="$HERMES_DIR/FIRST_RUN_REPORT.md"
    
    cat > "$REPORT_FILE" << EOF
# Hermes 

****: $(date +"%Y-%m-%d %H:%M:%S")

---

## 

 Hermes  AI 

---

## 

|  |  |
|------|-----|
|  | 1.0.0 |
| Gateway  | http://localhost:50051 |
|  | NanoGPT 12 |
|  GPU | 8 x NVIDIA H100 |

---

## 

1. ****: GatewayCheckpoint
2. ****:  REST API 
3. ****: 
4. ****: 

---

## 

|  |  |  |  |
|------|--------|--------|------|
|  | < 500ms | ⏳  | - |
|  | < 30s | ⏳  | - |
|  | < 5s | ⏳  | - |
| GPU | > 85% | ⏳  | - |

---

## 

### 

-  
-  
-  

### 

-  

---

## 

|  |  | Hermes |  |
|------|--------|--------|------|
|  | ~10s | ⏳  | - |
|  | ~120s | ⏳  | - |
| GPU | ~60% | ⏳  | - |

---

## 

...

---

## 

1. 
2. 
3. 
EOF
    
    log_ok ": $REPORT_FILE"
    cat "$REPORT_FILE"
}

# 
main() {
    echo "=============================================="
    echo " Hermes "
    echo "=============================================="
    echo ""
    
    check_dependencies
    echo ""
    
    start_services
    echo ""
    
    run_test_job
    echo ""
    
    generate_report
    
    echo ""
    echo "=============================================="
    echo " "
    echo "=============================================="
    echo ": $HERMES_DIR/FIRST_RUN_REPORT.md"
    echo ": $HERMES_DIR/hermes_job_result.json"
}

# 
case "${1:-all}" in
    all)
        main
        ;;
    dependencies)
        check_dependencies
        ;;
    services)
        start_services
        ;;
    job)
        run_test_job
        ;;
    report)
        generate_report
        ;;
    help|*)
        echo ": $0 {all|dependencies|services|job|report|help}"
        echo
        echo ":"
        echo "  all          - "
        echo "  dependencies - "
        echo "  services     - "
        echo "  job          - "
        echo "  report       - "
        echo "  help         - "
        ;;
esac
