#!/bin/bash
# Hermes 
# 

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
echo " Hermes "
echo "=============================================="
echo ""

# 1. Alertmanager
log_info "1. Alertmanager..."
if [ -f "deploy/prometheus/alertmanager.yml" ]; then
    log_ok ""
    
    # Webhook
    if grep -q "webhook_configs" "deploy/prometheus/alertmanager.yml"; then
        log_ok "Webhook"
    else
        log_err "Webhook"
        exit 1
    fi

    # 
    if grep -q "dingtalk" "deploy/prometheus/alertmanager.yml" || grep -q "oapi.dingtalk.com" "deploy/prometheus/alertmanager.yml"; then
        log_ok ""
    else
        log_warn ""
    fi
else
    log_err "Alertmanager"
    exit 1
fi

echo ""

# 2. 
log_info "2. ..."

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
        "summary": "",
        "description": ""
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
    "summary": "",
    "description": ""
  }
}
EOF
)

# 
echo "Alertmanager..."
curl -s -X POST http://localhost:9093/api/v2/alerts \
  -H "Content-Type: application/json" \
  -d "$TEST_ALERT" > /dev/null

if [ $? -eq 0 ]; then
    log_ok ""
else
    log_err ""
    exit 1
fi

echo ""

# 3. Alertmanager
log_info "3. Alertmanager..."
STATUS=$(curl -s http://localhost:9093/api/v2/status)
if echo "$STATUS" | grep -q "ready"; then
    log_ok "Alertmanager"
else
    log_err "Alertmanager"
    exit 1
fi

echo ""
echo "=============================================="
echo " !"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "   Alertmanager"
echo "   "
echo "   Alertmanager"
echo ""
echo " /"
echo ""
echo " :"
echo ""
echo "  1. Webhook URL"
echo "  2. alertmanager.ymlYOUR_ROBOT_TOKENtoken"
echo "  3. Alertmanager"
echo ""
