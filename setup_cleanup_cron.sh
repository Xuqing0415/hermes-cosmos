#!/bin/bash
# Hermes 
# cron6

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
echo "⏰ Hermes "
echo "=============================================="
echo ""

# 
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup.sh"
LOG_FILE="/var/log/hermes/cleanup.log"

# 1. 
log_info "1. ..."
if [ -f "$CLEANUP_SCRIPT" ]; then
    log_ok ": $CLEANUP_SCRIPT"
    chmod +x "$CLEANUP_SCRIPT"
else
    log_err ": $CLEANUP_SCRIPT"
    exit 1
fi

# 2. 
log_info "2. ..."
mkdir -p /var/log/hermes
log_ok ""

# 3. cron
log_info "3. cron..."

# cron
crontab -l > /tmp/crontab_backup 2>/dev/null || true

# cron
CRON_JOB="0 */6 * * * $CLEANUP_SCRIPT >> $LOG_FILE 2>&1"

# 
if crontab -l 2>/dev/null | grep -q "cleanup.sh"; then
    log_warn "cron..."
    # 
    crontab -l | grep -v "cleanup.sh" | crontab -
fi

# 
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

log_ok "cron: 6"

# 4. 
log_info "4. ..."
bash "$CLEANUP_SCRIPT" --dry-run 2>/dev/null || bash "$CLEANUP_SCRIPT"
log_ok ""

# 5. cron
log_info "5. cron..."
crontab -l | grep cleanup.sh
if [ $? -eq 0 ]; then
    log_ok "cron"
else
    log_err "cron"
    exit 1
fi

echo ""
echo "=============================================="
echo " !"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "   : $CLEANUP_SCRIPT"
echo "   : $LOG_FILE"
echo "  ⏰ : 6"
echo "   :"
echo "    - 24Checkpoint"
echo "    - "
echo "    - 3"
echo ""
echo " :"
echo ""
echo "  # "
echo "  bash $CLEANUP_SCRIPT"
echo ""
echo "  # "
echo "  cat $LOG_FILE"
echo ""
echo "  # cron"
echo "  crontab -l"
echo ""
