"""
Personalized Federated Learning Module

Implements Ditto and FedRep algorithms for personalized federated learning.
"""

from .ditto import DittoClient, DittoServer
from .fedrep import FedRepClient, FedRepServer
from .utils_pfl import (
    create_non_iid_cifar10,
    evaluate_personalized,
    compute_fairness_metrics
)

__all__ = [
    'DittoClient',
    'DittoServer',
    'FedRepClient',
    'FedRepServer',
    'create_non_iid_cifar10',
    'evaluate_personalized',
    'compute_fairness_metrics'
]
