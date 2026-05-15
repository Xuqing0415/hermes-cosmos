"""
Monitoring and Dashboard Module

Provides real-time monitoring and visualization for federated learning.
"""

from .dashboard import MonitoringDashboard, run_dashboard
from .prometheus_metrics import FederatedMetrics, MetricsCollector, metrics

__all__ = [
    'MonitoringDashboard',
    'run_dashboard',
    'FederatedMetrics',
    'MetricsCollector',
    'metrics'
]
