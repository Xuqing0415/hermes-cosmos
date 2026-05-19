"""
Federated Quantum Learning Module

Implements federated learning with quantum neural networks.
"""

from .fed_qnn import (
    QuantumCircuit,
    QuantumClassifier,
    FedQNNClient,
    FedQNNServer,
    FedQNNCoordinator,
    generate_quantum_dataset
)

__all__ = [
    'QuantumCircuit',
    'QuantumClassifier',
    'FedQNNClient',
    'FedQNNServer',
    'FedQNNCoordinator',
    'generate_quantum_dataset'
]