#!/bin/bash
# Hermes 每日健康检查脚本
# 每天凌晨自动运行，检查所有服务状态

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

echo "# Hermes 每日健康检查报告" > "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "**日期**: $DATE" >> "$LOG_DIR/$REPORT_FILE"
echo "**时间**: $(date +%H:%M:%S)" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"

# 检查Docker服务
echo "## 📦 Docker服务状态" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"

if systemctl is-active --quiet docker; then
    echo "| 服务 | 状态 |" >> "$LOG_DIR/$REPORT_FILE"
    echo "|------|------|" >> "$LOG_DIR/$REPORT_FILE"
    echo "| Docker | ✅ 运行中 |" >> "$LOG_DIR/$REPORT_FILE"
    log_ok "Docker 运行正常"
else
    echo "| Docker | ❌ 未运行 |" >> "$LOG_DIR/$REPORT_FILE"
    log_err "Docker 未运行"
fi

# 检查Hermes服务
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "## 🚀 Hermes服务状态" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "| 服务 | 状态 |" >> "$LOG_DIR/$REPORT_FILE"
echo "|------|------|" >> "$LOG_DIR/$REPORT_FILE"

services=("hermes-redis" "hermes-prometheus" "hermes-grafana" "hermes-jaeger" "hermes-scheduler" "hermes-gateway" "hermes-checkpoint")
failed_count=0

for service in "${services[@]}"; do
    status=$(docker inspect -f '{{.State.Status}}' "$service" 2>/dev/null || echo "unknown")
    if [ "$status" = "running" ]; then
        echo "| $service | ✅ 运行中 |" >> "$LOG_DIR/$REPORT_FILE"
        log_ok "$service 运行正常"
    else
        echo "| $service | ❌ $status |" >> "$LOG_DIR/$REPORT_FILE"
        log_err "$service 异常: $status"
        failed_count=$((failed_count + 1))
    fi
done

# 检查API
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "## 🔌 API健康检查" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "| API | 状态 | 延迟 |" >> "$LOG_DIR/$REPORT_FILE"
echo "|-----|------|------|" >> "$LOG_DIR/$REPORT_FILE"

# 调度器API
latency=$(curl -s -w '%{time_total}' -o /dev/null http://localhost:50051/health 2>/dev/null || echo "timeout")
if [ "$latency" != "timeout" ]; then
    echo "| 调度器 | ✅ 正常 | ${latency}s |" >> "$LOG_DIR/$REPORT_FILE"
    log_ok "调度器 API 正常"
else
    echo "| 调度器 | ❌ 超时 | - |" >> "$LOG_DIR/$REPORT_FILE"
    log_err "调度器 API 超时"
    failed_count=$((failed_count + 1))
fi

# Gateway API
latency=$(curl -s -w '%{time_total}' -o /dev/null http://localhost:8080/v1/health 2>/dev/null || echo "timeout")
if [ "$latency" != "timeout" ]; then
    echo "| Gateway | ✅ 正常 | ${latency}s |" >> "$LOG_DIR/$REPORT_FILE"
    log_ok "Gateway API 正常"
else
    echo "| Gateway | ❌ 超时 | - |" >> "$LOG_DIR/$REPORT_FILE"
    log_err "Gateway API 超时"
    failed_count=$((failed_count + 1))
fi

# 检查磁盘空间
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "## 💾 磁盘空间" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "| 挂载点 | 容量 | 已用 | 可用 | 使用率 |" >> "$LOG_DIR/$REPORT_FILE"
echo "|--------|------|------|------|--------|" >> "$LOG_DIR/$REPORT_FILE"

df -h --output=target,size,used,avail,pcent / /var/lib/docker 2>/dev/null | grep -v Filesystem | while read -r line; do
    echo "| $line |" >> "$LOG_DIR/$REPORT_FILE"
done

# 检查内存
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "## 🧠 内存使用" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"
mem_total=$(free -h | grep Mem | awk '{print $2}')
mem_used=$(free -h | grep Mem | awk '{print $3}')
mem_available=$(free -h | grep Mem | awk '{print $7}')
mem_usage=$(free | grep Mem | awk '{print $3/$2 * 100.0}' | cut -d. -f1)
echo "| 总内存 | 已用 | 可用 | 使用率 |" >> "$LOG_DIR/$REPORT_FILE"
echo "|--------|------|------|--------|" >> "$LOG_DIR/$REPORT_FILE"
echo "| $mem_total | $mem_used | $mem_available | ${mem_usage}% |" >> "$LOG_DIR/$REPORT_FILE"

# 总结
echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "## 📊 总结" >> "$LOG_DIR/$REPORT_FILE"
echo "" >> "$LOG_DIR/$REPORT_FILE"

if [ "$failed_count" -eq 0 ]; then
    echo "**状态**: ✅ 全部正常" >> "$LOG_DIR/$REPORT_FILE"
    echo "**建议**: 系统运行正常，继续监控" >> "$LOG_DIR/$REPORT_FILE"
    log_ok "✅ 健康检查全部通过"
else
    echo "**状态**: ❌ 有 $failed_count 项异常" >> "$LOG_DIR/$REPORT_FILE"
    echo "**建议**: 请检查异常服务" >> "$LOG_DIR/$REPORT_FILE"
    log_err "❌ 健康检查有 $failed_count 项异常"
fi

echo "" >> "$LOG_DIR/$REPORT_FILE"
echo "---" >> "$LOG_DIR/$REPORT_FILE"
echo "*自动生成于 $(date +%Y-%m-%d %H:%M:%S)*" >> "$LOG_DIR/$REPORT_FILE"

echo ""
echo "=============================================="
echo "📊 健康检查报告已生成"
echo "=============================================="
echo "报告位置: $LOG_DIR/$REPORT_FILE"
echo "异常数量: $failed_count"
echo ""

# 发送告警（如果有异常）
if [ "$failed_count" -gt 0 ]; then
    # 可以在这里添加邮件或钉钉告警
    echo "⚠️ 检测到异常，建议检查"
fi

exit $failed_count
