"""
Cross-Silo Federated Learning Module

Implements two-level federated learning for inter-organization collaboration.
"""

from .cross_silo_server import Silo, CrossSiloServer, CrossSiloCoordinator

__all__ = [
    'Silo',
    'CrossSiloServer',
    'CrossSiloCoordinator'
]
