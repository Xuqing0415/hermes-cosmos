#!/bin/bash
# Hermes 自动清理配置脚本
# 设置cron定时任务，每6小时清理一次

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[CRON]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "=============================================="
echo "⏰ Hermes 自动清理配置"
echo "=============================================="
echo ""

# 获取脚本绝对路径
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup.sh"
LOG_FILE="/var/log/hermes/cleanup.log"

# 1. 检查清理脚本是否存在
log_info "1. 检查清理脚本..."
if [ -f "$CLEANUP_SCRIPT" ]; then
    log_ok "清理脚本存在: $CLEANUP_SCRIPT"
    chmod +x "$CLEANUP_SCRIPT"
else
    log_err "清理脚本不存在: $CLEANUP_SCRIPT"
    exit 1
fi

# 2. 设置日志目录
log_info "2. 设置日志目录..."
mkdir -p /var/log/hermes
log_ok "日志目录已创建"

# 3. 设置cron任务
log_info "3. 设置cron任务..."

# 备份当前cron
crontab -l > /tmp/crontab_backup 2>/dev/null || true

# 添加新的cron任务
CRON_JOB="0 */6 * * * $CLEANUP_SCRIPT >> $LOG_FILE 2>&1"

# 检查是否已存在
if crontab -l 2>/dev/null | grep -q "cleanup.sh"; then
    log_warn "cron任务已存在，更新中..."
    # 删除旧任务
    crontab -l | grep -v "cleanup.sh" | crontab -
fi

# 添加新任务
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

log_ok "cron任务已设置: 每6小时执行一次"

# 4. 测试清理脚本
log_info "4. 测试清理脚本..."
bash "$CLEANUP_SCRIPT" --dry-run 2>/dev/null || bash "$CLEANUP_SCRIPT"
log_ok "清理脚本测试通过"

# 5. 验证cron配置
log_info "5. 验证cron配置..."
crontab -l | grep cleanup.sh
if [ $? -eq 0 ]; then
    log_ok "cron配置验证通过"
else
    log_err "cron配置失败"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ 自动清理配置完成!"
echo "=============================================="
echo ""
echo "📋 配置详情:"
echo ""
echo "  📁 清理脚本: $CLEANUP_SCRIPT"
echo "  📝 日志文件: $LOG_FILE"
echo "  ⏰ 执行频率: 每6小时"
echo "  🗑️ 清理规则:"
echo "    - 删除超过24小时的Checkpoint"
echo "    - 压缩日志文件"
echo "    - 保留最近3个版本"
echo ""
echo "💡 手动执行命令:"
echo ""
echo "  # 立即执行清理"
echo "  bash $CLEANUP_SCRIPT"
echo ""
echo "  # 查看清理日志"
echo "  cat $LOG_FILE"
echo ""
echo "  # 查看cron任务"
echo "  crontab -l"
echo ""
