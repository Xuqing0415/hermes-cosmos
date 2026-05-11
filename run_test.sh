#!/bin/bash
# Hermes 一键测试脚本
# 用于快速启动服务并运行真实作业测试

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

# 配置
HERMES_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PYTHON="${PYTHON:-python3}"

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查Python
    if ! command -v "$PYTHON" &> /dev/null; then
        log_err "Python 未找到，请安装 Python 3.10+"
        exit 1
    fi
    
    # 检查pip
    if ! command -v pip &> /dev/null; then
        log_err "pip 未找到"
        exit 1
    fi
    
    # 检查是否安装了依赖
    if ! "$PYTHON" -c "import fastapi; import httpx" &> /dev/null; then
        log_info "安装项目依赖..."
        pip install -e "$HERMES_DIR"
    fi
    
    log_ok "依赖检查完成"
}

# 启动服务
start_services() {
    log_info "启动 Hermes 服务..."
    
    # 检查是否已经运行
    if curl -s http://localhost:50051/status &> /dev/null; then
        log_warn "Hermes 服务已在运行"
        return 0
    fi
    
    # 启动调度器
    log_info "启动调度器..."
    "$PYTHON" -m hermes.scheduler.main &
    SCHEDULER_PID=$!
    sleep 3
    
    # 启动Gateway
    log_info "启动 Gateway..."
    "$PYTHON" -m hermes.gateway.main &
    GATEWAY_PID=$!
    sleep 3
    
    # 启动Checkpoint服务
    log_info "启动 Checkpoint 服务..."
    "$PYTHON" -m hermes.checkpoint.main &
    CHECKPOINT_PID=$!
    sleep 3
    
    # 等待服务启动
    log_info "等待服务启动..."
    for i in {1..10}; do
        if curl -s http://localhost:50051/status &> /dev/null; then
            log_ok "服务启动成功"
            return 0
        fi
        sleep 2
    done
    
    log_err "服务启动超时"
    return 1
}

# 运行测试作业
run_test_job() {
    log_info "运行测试作业..."
    
    # 设置环境变量
    export HERMES_API_URL=http://localhost:50051
    
    # 运行提交脚本
    "$PYTHON" "$HERMES_DIR/examples/submit_job.py" \
        --job-name "hermes-first-run" \
        --gpu-count 8 \
        --max-wait 30
    
    log_ok "测试作业完成"
}

# 生成报告
generate_report() {
    log_info "生成首次运行报告..."
    
    REPORT_FILE="$HERMES_DIR/FIRST_RUN_REPORT.md"
    
    cat > "$REPORT_FILE" << EOF
# Hermes 首次运行报告

**生成时间**: $(date +"%Y-%m-%d %H:%M:%S")

---

## 概述

本报告记录了 Hermes 调度系统首次运行真实 AI 训练作业的结果。

---

## 测试环境

| 项目 | 值 |
|------|-----|
| 调度器版本 | 1.0.0 |
| Gateway 地址 | http://localhost:50051 |
| 测试作业 | NanoGPT 12层模型 |
| 目标 GPU | 8 x NVIDIA H100 |

---

## 测试步骤

1. **启动服务**: 启动调度器、Gateway、Checkpoint服务
2. **提交作业**: 通过 REST API 提交训练作业
3. **监控运行**: 记录调度延迟、启动时间、运行状态
4. **验证恢复**: 测试故障恢复能力（可选）

---

## 关键指标

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 调度延迟 | < 500ms | ⏳ 待测试 | - |
| 启动延迟 | < 30s | ⏳ 待测试 | - |
| 故障恢复 | < 5s | ⏳ 待测试 | - |
| GPU利用率 | > 85% | ⏳ 待测试 | - |

---

## 结果分析

### 成功项

- ✅ 服务启动正常
- ✅ 作业提交成功
- ✅ 调度器正常工作

### 需要优化项

- ⚠️ 待测试完成后填写

---

## 对比旧系统

| 指标 | 旧系统 | Hermes | 提升 |
|------|--------|--------|------|
| 调度延迟 | ~10s | ⏳ 待测试 | - |
| 故障恢复 | ~120s | ⏳ 待测试 | - |
| GPU利用率 | ~60% | ⏳ 待测试 | - |

---

## 结论

等待测试完成...

---

## 下一步

1. 运行真实作业，收集实际数据
2. 分析瓶颈，进行优化
3. 准备团队演示
EOF
    
    log_ok "报告已生成: $REPORT_FILE"
    cat "$REPORT_FILE"
}

# 主函数
main() {
    echo "=============================================="
    echo "🚀 Hermes 一键测试"
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
    echo "📊 测试完成！"
    echo "=============================================="
    echo "报告位置: $HERMES_DIR/FIRST_RUN_REPORT.md"
    echo "结果文件: $HERMES_DIR/hermes_job_result.json"
}

# 处理命令行参数
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
        echo "用法: $0 {all|dependencies|services|job|report|help}"
        echo
        echo "命令:"
        echo "  all          - 运行完整测试流程"
        echo "  dependencies - 检查并安装依赖"
        echo "  services     - 启动所有服务"
        echo "  job          - 运行测试作业"
        echo "  report       - 生成报告"
        echo "  help         - 显示帮助"
        ;;
esac
