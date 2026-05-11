#!/bin/bash
# Hermes 下周行动执行脚本
# 规模化三部曲第一阶段：试点完成

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[WEEK2]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

HERMES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

echo "=============================================="
echo "🚀 Hermes 下周行动 - 规模化第一阶段"
echo "=============================================="
echo ""

# ========== 检查准备情况 ==========
log_info "检查准备情况..."

echo ""
echo "📋 复盘会议材料:"

if [ -f "$HERMES_DIR/RETROSPECTIVE_ONEPAGE.md" ]; then
    log_ok "  ✅ 一页摘要已准备"
else
    log_err "  ❌ 一页摘要未准备"
fi

if [ -f "$HERMES_DIR/FIRST_RUN_REPORT.md" ]; then
    log_ok "  ✅ 首次运行报告已准备"
else
    log_err "  ❌ 首次运行报告未准备"
fi

echo ""
echo "📋 用户支持材料:"

if [ -f "$HERMES_DIR/QUICK_START.md" ]; then
    log_ok "  ✅ 快速入门指南已准备"
else
    log_err "  ❌ 快速入门指南未准备"
fi

if [ -f "$HERMES_DIR/SLA_TEMPLATE.md" ]; then
    log_ok "  ✅ SLA模板已准备"
else
    log_err "  ❌ SLA模板未准备"
fi

# ========== 周一任务 ==========
echo ""
log_info "🎯 周一任务:"
echo ""
echo "  1. 📢 早会同步本周成果"
echo "     - 生产就绪检查50/50完成"
echo "     - 推理服务运行>24小时"
echo "     - 调度延迟350ms, GPU利用率87%"
echo ""
echo "  2. 📅 确认复盘会议时间"
echo "     - 建议时间：周三上午10点"
echo "     - 邀请：技术负责人、AI团队负责人、运维主管、产品经理"
echo ""
echo "  3. 🔔 将推理服务加入生产告警"
echo "     - 配置P99延迟告警"
echo "     - 配置GPU利用率告警"
echo "     - 设置值班轮转"

# ========== 周二任务 ==========
echo ""
log_info "🎯 周二任务:"
echo ""
echo "  1. 📊 准备复盘会议材料"
echo "     - 更新数据（推理服务运行满48小时）"
echo "     - 制作演示PPT"
echo ""
echo "  2. 📝 与第二个业务方签署SLA"
echo "     - 使用 SLA_TEMPLATE.md"
echo "     - 明确P99延迟<100ms"
echo ""
echo "  3. 👥 创建用户支持频道"
echo "     - 创建 #hermes-support Slack频道"
echo "     - 指定2名支持联系人"
echo "     - 发布快速入门指南"

# ========== 周三任务 ==========
echo ""
log_info "🎯 周三任务:"
echo ""
echo "  1. 🎬 复盘与立项扩大会议"
echo "     - 时间：上午10点"
echo "     - 目标：获得正式批准"
echo ""
echo "  2. 📋 根据会议决议更新计划"
echo "     - 记录批准事项"
echo "     - 更新资源申请"

# ========== 周四任务 ==========
echo ""
log_info "🎯 周四任务:"
echo ""
echo "  1. 📝 根据会议反馈调整资源申请"
echo ""
echo "  2. 🤝 联系第三个潜在业务"
echo "     - 作为有限推广备选"
echo ""
echo "  3. 📋 配置配额申请表单"
echo "     - Google Form"
echo "     - 收集作业特征、GPU需求、时间窗口"

# ========== 周五任务 ==========
echo ""
log_info "🎯 周五任务:"
echo ""
echo "  1. 📊 提交有限推广正式计划"
echo "     - 时间表"
echo "     - 成功标准"
echo ""
echo "  2. 📝 周报"
echo "     - 推理服务运行满7天"
echo "     - 无P0/P1事故"

# ========== 风险提醒 ==========
echo ""
log_info "⚠️ 风险提醒:"
echo ""
echo "  🔴 推理服务周末故障 - 安排值班或确保告警到人"
echo "  🔴 复盘会议质疑 - 准备详细数据，提前一对一沟通"
echo "  🔴 SLA条款争议 - 采用渐进式SLA（99.5%→99.7%→99.9%）"
echo "  🔴 支持通道无人响应 - 明确2小时响应承诺"

# ========== 完成 ==========
echo ""
echo "=============================================="
echo "✅ 下周行动计划已准备完成!"
echo "=============================================="
echo ""
echo "📋 关键产出:"
echo ""
echo "  ✅ RETROSPECTIVE_ONEPAGE.md - 复盘会议一页摘要"
echo "  ✅ QUICK_START.md - 用户快速入门指南"
echo "  ✅ SLA_TEMPLATE.md - 服务等级协议模板"
echo "  ✅ WEEKEND_WRAPUP.md - 周末收尾检查清单"
echo ""
echo "📅 重点日期:"
echo ""
echo "  🗓️ 周三 - 复盘与立项扩大会议"
echo "  🗓️ 周五 - 提交有限推广计划"
echo ""
echo "💡 明天行动:"
echo ""
echo "  1. 检查推理服务夜间运行日志"
echo "  2. 将一页摘要私信发给决策层关键人物"
echo "  3. 准备周一早会发言"
echo ""
