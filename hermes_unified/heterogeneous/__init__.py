"""
Heterogeneous Model Federated Learning Module

Enables federated learning with clients having different model architectures
using knowledge distillation and feature alignment.
"""

from .heterogeneous_fl import (
    TeacherModel,
    StudentCNN,
    StudentMobileNet,
    StudentMLP,
    HeterogeneousClient,
    HeterogeneousServer,
    create_heterogeneous_cluster
)

__all__ = [
    'TeacherModel',
    'StudentCNN',
    'StudentMobileNet',
    'StudentMLP',
    'HeterogeneousClient',
    'HeterogeneousServer',
    'create_heterogeneous_cluster'
]
