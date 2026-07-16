#!/bin/bash
# Hermes 

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[START]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }

echo "=============================================="
echo " Hermes AI - "
echo "=============================================="
echo ""

# 
log_info "1. ..."

if ! command -v python &> /dev/null; then
    log_err "PythonPython 3.10+"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    log_err "pip"
    exit 1
fi

log_ok "Python"

# 
log_info "2. /Python..."
pip install -q streamlit requests fastapi uvicorn httpx

log_ok ""

# 
mkdir -p logs

echo ""
echo "=============================================="
echo " "
echo "=============================================="
echo ""

# API
log_info " API Gateway ( 8000)..."
python -m uvicorn hermes.api_gateway.main:app --host 0.0.0.0 --port 8000 > logs/gateway.log 2>&1 &
GATEWAY_PID=$!
echo "   PID: $GATEWAY_PID"

# 
sleep 3

# 
log_info " Scheduler ( 8001)..."
python -m uvicorn hermes.scheduler.main:app --host 0.0.0.0 --port 8001 > logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "   PID: $SCHEDULER_PID"

# 
sleep 3

# Checkpoint
log_info " Checkpoint Service ( 8002)..."
python -m uvicorn hermes.checkpoint.server:app --host 0.0.0.0 --port 8002 > logs/checkpoint.log 2>&1 &
CHECKPOINT_PID=$!
echo "   PID: $CHECKPOINT_PID"

# 
sleep 3

# Streamlit UI
log_info " Web UI ( 8501)..."
echo ""
echo "=============================================="
echo " !"
echo "=============================================="
echo ""
echo " :"
echo ""
echo "   API Gateway:   http://localhost:8000"
echo "   Scheduler:     http://localhost:8001"
echo "   Checkpoint:    http://localhost:8002"
echo "   Web UI:        http://localhost:8501"
echo ""
echo "  Web UI "
echo ""

# PID
echo "$GATEWAY_PID" > logs/gateway.pid
echo "$SCHEDULER_PID" > logs/scheduler.pid
echo "$CHECKPOINT_PID" > logs/checkpoint.pid

# Streamlit
streamlit run ui/app.py
