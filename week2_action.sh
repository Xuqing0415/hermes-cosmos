#!/bin/bash
# Hermes 
# 

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
echo " Hermes  - "
echo "=============================================="
echo ""

# ==========  ==========
log_info "..."

echo ""
echo " :"

if [ -f "$HERMES_DIR/RETROSPECTIVE_ONEPAGE.md" ]; then
    log_ok "   "
else
    log_err "   "
fi

if [ -f "$HERMES_DIR/FIRST_RUN_REPORT.md" ]; then
    log_ok "   "
else
    log_err "   "
fi

echo ""
echo " :"

if [ -f "$HERMES_DIR/QUICK_START.md" ]; then
    log_ok "   "
else
    log_err "   "
fi

if [ -f "$HERMES_DIR/SLA_TEMPLATE.md" ]; then
    log_ok "   SLA"
else
    log_err "   SLA"
fi

# ==========  ==========
echo ""
log_info " :"
echo ""
echo "  1.  "
echo "     - 50/50"
echo "     - >24"
echo "     - 350ms, GPU87%"
echo ""
echo "  2.  "
echo "     - 10"
echo "     - AI"
echo ""
echo "  3.  "
echo "     - P99"
echo "     - GPU"
echo "     - "

# ==========  ==========
echo ""
log_info " :"
echo ""
echo "  1.  "
echo "     - 48"
echo "     - PPT"
echo ""
echo "  2.  SLA"
echo "     -  SLA_TEMPLATE.md"
echo "     - P99<100ms"
echo ""
echo "  3.  "
echo "     -  #hermes-support Slack"
echo "     - 2"
echo "     - "

# ==========  ==========
echo ""
log_info " :"
echo ""
echo "  1.  "
echo "     - 10"
echo "     - "
echo ""
echo "  2.  "
echo "     - "
echo "     - "

# ==========  ==========
echo ""
log_info " :"
echo ""
echo "  1.  "
echo ""
echo "  2.  "
echo "     - "
echo ""
echo "  3.  "
echo "     - Google Form"
echo "     - GPU"

# ==========  ==========
echo ""
log_info " :"
echo ""
echo "  1.  "
echo "     - "
echo "     - "
echo ""
echo "  2.  "
echo "     - 7"
echo "     - P0/P1"

# ==========  ==========
echo ""
log_info " :"
echo ""
echo "    - "
echo "    - "
echo "   SLA - SLA99.5%→99.7%→99.9%"
echo "    - 2"

# ==========  ==========
echo ""
echo "=============================================="
echo " !"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "   RETROSPECTIVE_ONEPAGE.md - "
echo "   QUICK_START.md - "
echo "   SLA_TEMPLATE.md - "
echo "   WEEKEND_WRAPUP.md - "
echo ""
echo " :"
echo ""
echo "    - "
echo "    - "
echo ""
echo " :"
echo ""
echo "  1. "
echo "  2. "
echo "  3. "
echo ""
