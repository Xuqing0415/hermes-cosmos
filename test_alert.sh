#!/bin/bash
# Hermes 告警测试脚本
# 用于验证告警配置是否正确

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[ALERT]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "=============================================="
echo "🔔 Hermes 告警配置测试"
echo "=============================================="
echo ""

# 1. 检查Alertmanager配置
log_info "1. 检查Alertmanager配置..."
if [ -f "deploy/prometheus/alertmanager.yml" ]; then
    log_ok "配置文件存在"
    
    # 检查Webhook配置
    if grep -q "webhook_configs" "deploy/prometheus/alertmanager.yml"; then
        log_ok "Webhook配置已设置"
    else
        log_err "Webhook配置缺失"
        exit 1
    fi

    # 检查钉钉配置
    if grep -q "dingtalk" "deploy/prometheus/alertmanager.yml" || grep -q "oapi.dingtalk.com" "deploy/prometheus/alertmanager.yml"; then
        log_ok "钉钉告警配置已设置"
    else
        log_warn "钉钉配置未找到，请确认配置"
    fi
else
    log_err "Alertmanager配置文件不存在"
    exit 1
fi

echo ""

# 2. 发送测试告警
log_info "2. 发送测试告警..."

TEST_ALERT=$(cat <<EOF
{
  "receiver": "dingding",
  "status": "firing",
  "alerts": [
    {
      "status": "firing",
      "labels": {
        "alertname": "TestAlert",
        "severity": "critical",
        "instance": "hermes-test"
      },
      "annotations": {
        "summary": "测试告警",
        "description": "这是一个测试告警，请确认收到"
      },
      "startsAt": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    }
  ],
  "groupLabels": {"alertname": "TestAlert"},
  "commonLabels": {
    "alertname": "TestAlert",
    "severity": "critical",
    "instance": "hermes-test"
  },
  "commonAnnotations": {
    "summary": "测试告警",
    "description": "这是一个测试告警，请确认收到"
  }
}
EOF
)

# 发送测试告警
echo "发送测试告警到Alertmanager..."
curl -s -X POST http://localhost:9093/api/v2/alerts \
  -H "Content-Type: application/json" \
  -d "$TEST_ALERT" > /dev/null

if [ $? -eq 0 ]; then
    log_ok "测试告警发送成功"
else
    log_err "测试告警发送失败"
    exit 1
fi

echo ""

# 3. 验证Alertmanager状态
log_info "3. 验证Alertmanager状态..."
STATUS=$(curl -s http://localhost:9093/api/v2/status)
if echo "$STATUS" | grep -q "ready"; then
    log_ok "Alertmanager运行正常"
else
    log_err "Alertmanager状态异常"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ 告警测试完成!"
echo "=============================================="
echo ""
echo "📋 测试结果:"
echo ""
echo "  ✅ Alertmanager配置检查通过"
echo "  ✅ 测试告警已发送"
echo "  ✅ Alertmanager运行正常"
echo ""
echo "💡 请检查你的钉钉/邮件，确认收到测试告警"
echo ""
echo "📝 下一步:"
echo ""
echo "  1. 如果未收到告警，请检查Webhook URL是否正确"
echo "  2. 修改alertmanager.yml中的YOUR_ROBOT_TOKEN为实际token"
echo "  3. 重启Alertmanager使配置生效"
echo ""
