# Hermes Federated Learning - Production Deployment Guide

This guide covers deploying Hermes Federated Learning system to production using Docker, Kubernetes, and Helm.

## Prerequisites

- Docker 20.10+
- Kubernetes 1.24+ (or kind/minikube for local development)
- Helm 3.10+
- kubectl configured with cluster access

## Quick Start

### 1. One-Click Deployment (Local Cluster)

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

This will:
- Check prerequisites
- Create a local Kubernetes cluster (using kind if available)
- Build Docker image
- Install Helm chart
- Apply configurations
- Verify deployment

### 2. Docker Compose (Development)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f hermes-server

# Stop all services
docker-compose down
```

Access:
- Server: http://localhost:8000
- Dashboard: http://localhost:8501
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (admin/admin)

## Kubernetes Deployment

### Manual Deployment

```bash
# Create namespace
kubectl apply -f kubernetes/namespace.yaml

# Apply configurations
kubectl apply -f kubernetes/configmap.yaml -n hermes

# Deploy server
kubectl apply -f kubernetes/server-deployment.yaml -n hermes

# Deploy clients
kubectl apply -f kubernetes/client-deployment.yaml -n hermes

# Apply HPA (auto-scaling)
kubectl apply -f kubernetes/hpa.yaml -n hermes

# Check status
kubectl get pods -n hermes
```

### Using Helm

```bash
# Install chart
helm install hermes kubernetes/helm/hermes -n hermes --create-namespace

# Upgrade
helm upgrade hermes kubernetes/helm/hermes -n hermes

# Uninstall
helm uninstall hermes -n hermes
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of server replicas | 1 |
| `server.config.num_rounds` | Federated training rounds | 100 |
| `server.config.clients_per_round` | Clients per round | 10 |
| `server.config.defense_type` | Defense mechanism | krum |
| `client.replicaCount` | Number of client pods | 20 |
| `monitoring.enabled` | Enable Prometheus metrics | true |

## Monitoring

### Prometheus Metrics

Hermes exposes the following metrics at `:9090/metrics`:

- `hermes_round_duration_seconds` - Time for each round
- `hermes_global_accuracy` - Current model accuracy
- `hermes_active_clients_count` - Active clients per round
- `hermes_communication_bytes_total` - Total bytes communicated
- `hermes_attack_events_total` - Detected attacks
- `hermes_defense_effectiveness` - Defense effectiveness score

### Grafana Dashboard

Import `grafana/provisioning/dashboards/hermes-dashboard.json` into Grafana for pre-built visualizations.

### Alerts

Prometheus alerts are configured in `kubernetes/prometheus-service-monitor.yaml`:

- `HermesAccuracyDrop` - Accuracy drops > 5%
- `HermesLowParticipation` - Participation < 50%
- `HermesHighLatency` - Round duration > 2 minutes
- `HermesServerDown` - Server unreachable

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yaml`) provides:

1. **Unit Tests** - Run pytest on all modules
2. **Integration Tests** - Test federated training
3. **Docker Build** - Build and push images
4. **Kubernetes Deploy** - Deploy to cluster

### Setup Secrets

```bash
# Required secrets in GitHub repository
DOCKER_HUB_TOKEN    # Docker Hub access token
KUBE_CONFIG         # Base64-encoded kubectl config
```

## Production Checklist

- [ ] Enable TLS/SSL for all endpoints
- [ ] Configure persistent storage volumes
- [ ] Set up proper resource limits
- [ ] Configure alerting notifications
- [ ] Enable log aggregation
- [ ] Set up backup strategy
- [ ] Configure network policies
- [ ] Enable audit logging

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n hermes
kubectl describe pod <pod-name> -n hermes
kubectl logs <pod-name> -n hermes
```

### Common Issues

**Pod stuck in CrashLoopBackOff**
- Check logs: `kubectl logs <pod-name> -n hermes`
- Verify ConfigMap is correctly applied
- Check resource limits

**Clients not connecting**
- Verify server service is running
- Check client configuration (SERVER_URL)
- Check network policies

**HPA not scaling**
- Verify metrics-server is installed
- Check HPA configuration
- Review resource utilization

## Architecture

```

                     Kubernetes Cluster                   
                                                         
           
     Server           Client           Client     
    (hermes)      Pod 1            Pod N      
                                                  
           
                                                     
                         
                                                      
                                                      
                          
   Prometheus                        Grafana        
                                                    
                          
                                                         

```

## Support

For issues and questions, please open an issue on the GitHub repository.
