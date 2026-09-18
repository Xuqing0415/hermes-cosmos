#!/bin/bash
# 快速启动单个组件进行开发和测试
# 用法: ./dev.sh [gateway|scheduler|checkpoint|agent|all]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[DEV]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_err() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_err "Python 3 未安装"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # 检查虚拟环境
    if [ ! -d "venv" ]; then
        log_info "创建虚拟环境..."
        python3 -m venv venv
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 检查依赖
    if ! python -c "import hermes" 2> /dev/null; then
        log_info "安装依赖..."
        pip install -e ".[dev]"
    fi
}

# 启动调度器
start_scheduler() {
    check_python
    log_info "启动调度器 (端口 50051)..."
    cd "$PROJECT_ROOT"
    python -m hermes.scheduler.main &
    SCHEDULER_PID=$!
    echo $SCHEDULER_PID > .scheduler.pid
    log_ok "调度器已启动 (PID: $SCHEDULER_PID)"
}

# 启动网关
start_gateway() {
    check_python
    log_info "启动API网关 (端口 8080)..."
    cd "$PROJECT_ROOT"
    python -m hermes.gateway.main &
    GATEWAY_PID=$!
    echo $GATEWAY_PID > .gateway.pid
    log_ok "网关已启动 (PID: $GATEWAY_PID)"
}

# 测试作业提交
submit_test_job() {
    check_python
    log_info "提交测试作业..."
    
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

print("响应状态码:", response.status_code)
print(json.dumps(response.json(), indent=2))
EOF
}

# 运行端到端测试
run_e2e_tests() {
    check_python
    log_info "运行端到端测试..."
    cd "$PROJECT_ROOT"
    pytest tests/test_e2e.py -v -s
}

# 运行单元测试
run_unit_tests() {
    check_python
    log_info "运行单元测试..."
    cd "$PROJECT_ROOT"
    pytest tests/test_core.py tests/test_scheduler.py tests/test_checkpoint.py -v
}

# 停止所有服务
stop_all() {
    log_info "停止所有服务..."
    
    for pid_file in .scheduler.pid .gateway.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file" 2>/dev/null || true)
            if [ -n "$pid" ]; then
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    
    # 杀掉所有相关进程
    pkill -f "hermes.*\.main" 2>/dev/null || true
    
    log_ok "所有服务已停止"
}

# 主菜单
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
            log_ok "环境已清理"
            ;;
        help|*)
            echo "Hermes 开发工具"
            echo
            echo "用法: $0 [命令]"
            echo
            echo "命令:"
            echo "  scheduler - 启动调度器"
            echo "  gateway   - 启动API网关"
            echo "  all       - 启动所有服务"
            echo "  test-job  - 提交测试作业"
            echo "  unit      - 运行单元测试"
            echo "  e2e       - 运行端到端测试"
            echo "  test      - 运行所有测试"
            echo "  stop      - 停止所有服务"
            echo "  clean     - 清理环境"
            echo
            ;;
    esac
}

# 清理函数
cleanup() {
    log_info "收到终止信号，清理中..."
    stop_all
    exit 0
}

trap cleanup SIGINT SIGTERM

main "$@"
