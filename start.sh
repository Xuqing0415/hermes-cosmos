#!/bin/bash
# Hermes Cosmos - 
# 

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"

# 
check_dependencies() {
    log_info "..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker  Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        log_error "Docker Compose  Docker Compose"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3  Python 3.11+"
        exit 1
    fi
    
    log_info ""
}

# 
start_services() {
    log_info " Hermes ..."
    
    cd "$PROJECT_ROOT/docker"
    
    # 
    docker-compose up -d postgres redis etcd
    log_info "..."
    sleep 15
    
    # 
    docker-compose up -d prometheus grafana jaeger
    sleep 5
    
    # 
    docker-compose up -d gateway scheduler checkpoint agent
    
    log_info ""
}

# 
wait_for_services() {
    log_info "..."
    
    #  Gateway
    local timeout=60
    local count=0
    while [ $count -lt $timeout ]; do
        if curl -s "http://localhost:8080/v1/health" > /dev/null 2>&1; then
            log_info "Gateway "
            break
        fi
        sleep 1
        count=$((count + 1))
    done
    
    if [ $count -eq $timeout ]; then
        log_warn "Gateway "
    fi
}

# 
health_check() {
    log_info "..."
    
    local report_file="$PROJECT_ROOT/HEALTH_CHECK_REPORT.md"
    cat > "$report_file" << EOF
# Hermes Cosmos 

****: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
****: 
****: $(grep -m 1 version "$PROJECT_ROOT/pyproject.toml" | cut -d'"' -f2)

---

## 

|  |  |  |  |
|------|------|------|---------|
EOF
    
    local services=(
        "Gateway:http://localhost:8080/v1/health"
        "Prometheus:http://localhost:9090/-/healthy"
        "Grafana:http://localhost:3000/api/health"
        "Jaeger:http://localhost:16686"
    )
    
    for service_entry in "${services[@]}"; do
        IFS=':' read -r name url <<< "$service_entry"
        
        local status=" "
        local response_time="-"
        
        if [ "$(curl -s -o /dev/null -w "%{http_code}" "$url")" = "200" ]; then
            status=" "
            response_time="$(curl -s -o /dev/null -w "%{time_total}" "$url")s"
        fi
        
        echo "| $name | $url | $status | $response_time |" >> "$report_file"
    done
    
    # 
    echo -e "\n---\n\n## \n" >> "$report_file"
    echo '```' >> "$report_file"
    docker-compose -f "$COMPOSE_FILE" ps --format table >> "$report_file"
    echo '```' >> "$report_file"
    
    log_info ": $report_file"
}

# 
show_logs() {
    cd "$PROJECT_ROOT/docker"
    docker-compose logs -f
}

# 
stop_services() {
    log_info " Hermes ..."
    cd "$PROJECT_ROOT/docker"
    docker-compose down
}

# 
cleanup() {
    log_warn ""
    read -p "(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_ROOT/docker"
        docker-compose down -v
        log_info ""
    fi
}

# 
main() {
    case "${1:-start}" in
        start)
            check_dependencies
            start_services
            wait_for_services
            health_check
            echo
            log_info ""
            log_info "Gateway: http://localhost:8080"
            log_info "Grafana: http://localhost:3000 (admin/hermes2026)"
            log_info "Prometheus: http://localhost:9090"
            log_info "Jaeger: http://localhost:16686"
            log_info
            log_info ": $0 logs"
            log_info ": $0 stop"
            log_info ": HEALTH_CHECK_REPORT.md"
            ;;
        stop)
            stop_services
            ;;
        logs)
            show_logs
            ;;
        health)
            health_check
            ;;
        cleanup)
            cleanup
            ;;
        *)
            echo ": $0 {start|stop|logs|health|cleanup}"
            echo
            echo ""
            echo "  start   - "
            echo "  stop    - "
            echo "  logs    - "
            echo "  health  - "
            echo "  cleanup - "
            exit 1
            ;;
    esac
}

main "$@"
