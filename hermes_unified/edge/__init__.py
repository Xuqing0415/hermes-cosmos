"""
Cloud-Edge Federated Learning Module

Implements simulated edge devices, cloud server, and adaptive model compression
for cloud-edge federated learning research.
"""

from .edge_simulator import SimulatedEdgeDevice, EdgeDeviceFactory
from .cloud_server import CloudServer, CloudEdgeCoordinator
from .pruning import prune_model_by_speed, prune_by_ratio, quantize_model, AdaptiveModelManager

__all__ = [
    'SimulatedEdgeDevice',
    'EdgeDeviceFactory',
    'CloudServer',
    'CloudEdgeCoordinator',
    'prune_model_by_speed',
    'prune_by_ratio',
    'quantize_model',
    'AdaptiveModelManager'
]
