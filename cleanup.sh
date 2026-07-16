#!/bin/bash
# Hermes 
# CheckpointPod

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[CLEANUP]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "=============================================="
echo " Hermes "
echo "=============================================="
echo ""

# 1. Checkpoint
log_info "1. Checkpoint..."
CHECKPOINT_DIR="/var/lib/hermes/checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    # 7Checkpoint
    find "$CHECKPOINT_DIR" -type f -mtime +7 -delete
    log_ok "7Checkpoint"
else
    log_warn "Checkpoint"
fi

# 2. 
log_info "2. ..."
LOG_DIR="/var/log/hermes"
if [ -d "$LOG_DIR" ]; then
    find "$LOG_DIR" -type f -name "*.log" -mtime +7 -delete
    log_ok "7"
else
    log_warn ""
fi

# 3. Docker
log_info "3. ..."
docker ps -q --filter "status=exited" | xargs -r docker rm 2>/dev/null || log_warn ""
log_ok ""

# 4. 
read -p "Docker? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker image prune -f
    log_ok ""
fi

# 5. 
log_info "4. ..."

services=("hermes-scheduler" "hermes-gateway" "hermes-checkpoint")
for service in "${services[@]}"; do
    status=$(docker inspect -f '{{.State.Status}}' "$service" 2>/dev/null || echo "not found")
    if [ "$status" != "running" ]; then
        log_warn "$service ..."
        docker start "$service" 2>/dev/null || log_err ""
    else
        log_ok "$service "
    fi
done

# 6. 
log_info "5. ..."
rm -rf /tmp/hermes_*.tmp 2>/dev/null || true
log_ok ""

echo ""
echo "=============================================="
echo " !"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "   7Checkpoint"
echo "   7"
echo "   Docker"
echo "   "
echo "   "
echo ""
echo " "
echo ""
