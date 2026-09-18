#!/bin/bash
# Hermes 运营监控脚本
# 用于每日/每周自动检查系统状态

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

# 配置
API_URL="${HERMES_API_URL:-http://localhost:50051}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"  # 5分钟检查一次
ALERT_THRESHOLD="${ALERT_THRESHOLD:-0.85}"  # 85%阈值

# 告警函数
send_alert() {
    local level=$1
    local message=$2
    echo "[$(date)] [$level] $message" >> /var/log/hermes/monitor.log
    
    # 可以在这里添加Slack/邮件告警
    # curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"$message\"}" "$SLACK_WEBHOOK"
}

# 检查调度器状态
check_scheduler() {
    log_info "检查调度器状态..."
    
    response=$(curl -s "$API_URL/status")
    
    if [ -z "$response" ]; then
        log_err "调度器无响应"
        send_alert "CRITICAL" "调度器无响应"
        return 1
    fi
    
    status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    
    if [ "$status" != "running" ]; then
        log_err "调度器状态异常: $status"
        send_alert "ERROR" "调度器状态异常: $status"
        return 1
    fi
    
    pending=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['pending_jobs'])")
    running=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['running_jobs'])")
    
    log_ok "调度器运行正常"
    log_info "  - Pending jobs: $pending"
    log_info "  - Running jobs: $running"
    
    return 0
}

# 检查集群资源
check_cluster() {
    log_info "检查集群资源..."
    
    response=$(curl -s "$API_URL/cluster/summary")
    
    if [ -z "$response" ]; then
        log_err "集群状态无响应"
        return 1
    fi
    
    total=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_gpus'])")
    available=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['available_gpus'])")
    
    utilization=$(( (total - available) * 100 / total ))
    
    log_info "  - 总GPU: $total"
    log_info "  - 可用GPU: $available"
    log_info "  - 利用率: ${utilization}%"
    
    if [ $utilization -gt 95 ]; then
        log_warn "GPU利用率过高 (${utilization}%)"
        send_alert "WARNING" "GPU利用率过高: ${utilization}%"
    fi
    
    if [ $available -lt 10 ]; then
        log_warn "可用GPU不足 (仅 $available 个)"
        send_alert "WARNING" "可用GPU不足: $available"
    fi
    
    return 0
}

# 检查故障预测
check_predictions() {
    log_info "检查故障预测..."
    
    # 这里应该调用Agent的预测接口
    log_info "  - 暂无高置信度故障预测"
    
    return 0
}

# 生成每日报告
generate_daily_report() {
    log_info "生成每日报告..."
    
    REPORT_DIR="/var/log/hermes/reports"
    mkdir -p "$REPORT_DIR"
    
    REPORT_FILE="$REPORT_DIR/daily_report_$(date +%Y%m%d).md"
    
    cat > "$REPORT_FILE" << EOF
# Hermes 每日运营报告

**日期**: $(date +"%Y年%m月%d日")

## 系统状态

\`\`\`
$(curl -s "$API_URL/status" | python3 -m json.tool)
\`\`\`

## 集群资源

\`\`\`
$(curl -s "$API_URL/cluster/summary" | python3 -m json.tool)
\`\`\`

## 作业统计

### 今日作业
- 提交数量: 0
- 完成数量: 0
- 失败数量: 0

### 运行中作业
- 数量: 0

## 告警汇总

### 警告
- 无

### 错误
- 无

## 建议

- 暂无特殊建议
EOF
    
    log_ok "每日报告已生成: $REPORT_FILE"
}

# 主循环
main() {
    log_info "启动 Hermes 运营监控..."
    
    # 确保日志目录存在
    mkdir -p /var/log/hermes
    
    while true; do
        echo ""
        log_info "=== 检查周期: $(date) ==="
        
        check_scheduler
        check_cluster
        check_predictions
        
        # 每日报告（早上8点）
        HOUR=$(date +%H)
        if [ "$HOUR" = "08" ]; then
            generate_daily_report
            # 避免重复生成
            sleep 3600
        fi
        
        # 每周报告（周一）
        DAY=$(date +%u)
        if [ "$DAY" = "1" ] && [ "$HOUR" = "09" ]; then
            log_info "生成每周报告..."
            # 这里可以添加周报逻辑
            sleep 3600
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# 处理命令行参数
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
        echo "用法: $0 {monitor|report|check|help}"
        echo
        echo "命令:"
        echo "  monitor - 启动持续监控"
        echo "  report  - 生成每日报告"
        echo "  check   - 执行一次检查"
        echo "  help    - 显示帮助"
        ;;
esac
