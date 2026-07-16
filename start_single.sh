#!/bin/bash
# Hermes 

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[SETUP]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

HERMES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

echo "=============================================="
echo " Hermes "
echo "=============================================="
echo ""

# 
log_info "..."

if ! command -v docker &> /dev/null; then
    log_err "Docker  Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_err "Docker Compose "
    exit 1
fi

log_ok "Docker  Docker Compose "
echo ""

# 
log_info " Hermes ..."

cd "$HERMES_DIR"

docker-compose -f docker-compose-single.yml build

log_ok ""
echo ""

# 
log_info "..."

docker-compose -f docker-compose-single.yml up -d

log_ok "..."
echo ""

# 
log_info "30..."
sleep 30

# 
log_info "..."
echo ""

services=("hermes-redis" "hermes-prometheus" "hermes-grafana" "hermes-jaeger" "hermes-scheduler" "hermes-gateway" "hermes-checkpoint")

for service in "${services[@]}"; do
    status=$(docker inspect -f '{{.State.Status}}' "$service" 2>/dev/null || echo "not found")
    if [ "$status" = "running" ]; then
        echo " $service"
    else
        echo " $service ($status)"
    fi
done

echo ""

# 
log_info " API..."
if curl -s http://localhost:50051/health &>/dev/null; then
    log_ok " API "
else
    log_err " API "
fi

#  Gateway
log_info " Gateway API..."
if curl -s http://localhost:8080/v1/health &>/dev/null; then
    log_ok "Gateway API "
else
    log_err "Gateway API "
fi

echo ""
echo "=============================================="
echo " !"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "   : http://localhost:50051"
echo "   Gateway: http://localhost:8080"
echo "   Grafana: http://localhost:3000 (admin/hermes)"
echo "   Prometheus: http://localhost:9090"
echo "   Jaeger: http://localhost:16686"
echo ""
echo " :"
echo ""
echo "  # "
echo "  curl -X POST http://localhost:50051/jobs \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"name\":\"test-job\",\"tenant_id\":\"test\",\"user_id\":\"test\",\"requirements\":{\"gpu_count\":1},\"image\":\"pytorch/pytorch:latest\"}'"
echo ""
echo "  # "
echo "  curl http://localhost:50051/jobs"
echo ""
echo "  # Pod"
echo "  docker kill hermes-scheduler"
echo "  # 5"
echo ""
echo " :"
echo ""
echo "  - Redis: ./data/redis"
echo "  - Prometheus: ./data/prometheus"
echo "  - Grafana: ./data/grafana"
echo "  - Checkpoint: ./data/checkpoints"
echo ""
