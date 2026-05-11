#!/bin/bash
# Hermes Cosmos - 启动脚本
# 用于本地开发和测试环境

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

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 未安装，请先安装 Python 3.11+"
        exit 1
    fi
    
    log_info "依赖检查通过"
}

# 启动服务
start_services() {
    log_info "启动 Hermes 服务..."
    
    cd "$PROJECT_ROOT/docker"
    
    # 启动基础设施
    docker-compose up -d postgres redis etcd
    log_info "等待基础设施就绪..."
    sleep 15
    
    # 启动可观测性服务
    docker-compose up -d prometheus grafana jaeger
    sleep 5
    
    # 启动核心服务
    docker-compose up -d gateway scheduler checkpoint agent
    
    log_info "服务启动完成！"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."
    
    # 等待 Gateway
    local timeout=60
    local count=0
    while [ $count -lt $timeout ]; do
        if curl -s "http://localhost:8080/v1/health" > /dev/null 2>&1; then
            log_info "Gateway 已就绪"
            break
        fi
        sleep 1
        count=$((count + 1))
    done
    
    if [ $count -eq $timeout ]; then
        log_warn "Gateway 未在预期时间内就绪，请检查日志"
    fi
}

# 健康检查
health_check() {
    log_info "执行健康检查..."
    
    local report_file="$PROJECT_ROOT/HEALTH_CHECK_REPORT.md"
    cat > "$report_file" << EOF
# Hermes Cosmos 健康检查报告

**生成时间**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**环境**: 本地开发环境
**版本**: $(grep -m 1 version "$PROJECT_ROOT/pyproject.toml" | cut -d'"' -f2)

---

## 服务状态

| 服务 | 地址 | 状态 | 响应时间 |
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
        
        local status="❌ 未健康"
        local response_time="-"
        
        if [ "$(curl -s -o /dev/null -w "%{http_code}" "$url")" = "200" ]; then
            status="✅ 健康"
            response_time="$(curl -s -o /dev/null -w "%{time_total}" "$url")s"
        fi
        
        echo "| $name | $url | $status | $response_time |" >> "$report_file"
    done
    
    # 容器状态
    echo -e "\n---\n\n## 容器状态\n" >> "$report_file"
    echo '```' >> "$report_file"
    docker-compose -f "$COMPOSE_FILE" ps --format table >> "$report_file"
    echo '```' >> "$report_file"
    
    log_info "健康检查报告已生成: $report_file"
}

# 查看日志
show_logs() {
    cd "$PROJECT_ROOT/docker"
    docker-compose logs -f
}

# 停止服务
stop_services() {
    log_info "停止 Hermes 服务..."
    cd "$PROJECT_ROOT/docker"
    docker-compose down
}

# 清理数据
cleanup() {
    log_warn "警告：这将删除所有数据！"
    read -p "确认要继续吗？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_ROOT/docker"
        docker-compose down -v
        log_info "数据清理完成"
    fi
}

# 主函数
main() {
    case "${1:-start}" in
        start)
            check_dependencies
            start_services
            wait_for_services
            health_check
            echo
            log_info "服务已启动！"
            log_info "Gateway: http://localhost:8080"
            log_info "Grafana: http://localhost:3000 (admin/hermes2026)"
            log_info "Prometheus: http://localhost:9090"
            log_info "Jaeger: http://localhost:16686"
            log_info
            log_info "查看日志: $0 logs"
            log_info "停止服务: $0 stop"
            log_info "健康检查报告: HEALTH_CHECK_REPORT.md"
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
            echo "用法: $0 {start|stop|logs|health|cleanup}"
            echo
            echo "命令说明："
            echo "  start   - 启动所有服务"
            echo "  stop    - 停止所有服务"
            echo "  logs    - 查看实时日志"
            echo "  health  - 执行健康检查"
            echo "  cleanup - 清理所有数据"
            exit 1
            ;;
    esac
}

main "$@"
