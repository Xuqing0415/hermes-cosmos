#!/bin/bash
# 
# : ./dev.sh [gateway|scheduler|checkpoint|agent|all]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[DEV]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_err() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_err "Python 3 "
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # 
    if [ ! -d "venv" ]; then
        log_info "..."
        python3 -m venv venv
    fi
    
    # 
    source venv/bin/activate
    
    # 
    if ! python -c "import hermes" 2> /dev/null; then
        log_info "..."
        pip install -e ".[dev]"
    fi
}

# 
start_scheduler() {
    check_python
    log_info " ( 50051)..."
    cd "$PROJECT_ROOT"
    python -m hermes.scheduler.main &
    SCHEDULER_PID=$!
    echo $SCHEDULER_PID > .scheduler.pid
    log_ok " (PID: $SCHEDULER_PID)"
}

# 
start_gateway() {
    check_python
    log_info "API ( 8080)..."
    cd "$PROJECT_ROOT"
    python -m hermes.gateway.main &
    GATEWAY_PID=$!
    echo $GATEWAY_PID > .gateway.pid
    log_ok " (PID: $GATEWAY_PID)"
}

# 
submit_test_job() {
    check_python
    log_info "..."
    
    python3 << 'EOF'
import json
import httpx

response = httpx.post(
    "http://localhost:50051/jobs",
    json={
        "name": "test-job",
        "tenant_id": "tenant-1",
        "user_id": "user-1",
        "priority": "HIGH",
        "requirements": {
            "gpu_count": 8,
            "gpu_type": "nvidia-h100",
            "memory_gb": 128,
            "cpu_cores": 32
        },
        "image": "pytorch/pytorch:latest",
        "command": "python train.py"
    }
)

print(":", response.status_code)
print(json.dumps(response.json(), indent=2))
EOF
}

# 
run_e2e_tests() {
    check_python
    log_info "..."
    cd "$PROJECT_ROOT"
    pytest tests/test_e2e.py -v -s
}

# 
run_unit_tests() {
    check_python
    log_info "..."
    cd "$PROJECT_ROOT"
    pytest tests/test_core.py tests/test_scheduler.py tests/test_checkpoint.py -v
}

# 
stop_all() {
    log_info "..."
    
    for pid_file in .scheduler.pid .gateway.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file" 2>/dev/null || true)
            if [ -n "$pid" ]; then
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    
    # 
    pkill -f "hermes.*\.main" 2>/dev/null || true
    
    log_ok ""
}

# 
main() {
    case "${1:-help}" in
        scheduler)
            start_scheduler
            wait
            ;;
        gateway)
            start_gateway
            wait
            ;;
        all)
            start_scheduler
            sleep 2
            start_gateway
            wait
            ;;
        test-job)
            submit_test_job
            ;;
        e2e)
            run_e2e_tests
            ;;
        unit)
            run_unit_tests
            ;;
        test)
            run_unit_tests && run_e2e_tests
            ;;
        stop)
            stop_all
            ;;
        clean)
            stop_all
            rm -rf venv __pycache__ .pytest_cache
            log_ok ""
            ;;
        help|*)
            echo "Hermes "
            echo
            echo ": $0 []"
            echo
            echo ":"
            echo "  scheduler - "
            echo "  gateway   - API"
            echo "  all       - "
            echo "  test-job  - "
            echo "  unit      - "
            echo "  e2e       - "
            echo "  test      - "
            echo "  stop      - "
            echo "  clean     - "
            echo
            ;;
    esac
}

# 
cleanup() {
    log_info "..."
    stop_all
    exit 0
}

trap cleanup SIGINT SIGTERM

main "$@"
