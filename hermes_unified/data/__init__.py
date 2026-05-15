"""
Data Module for Real Federated Datasets

Supports FEMNIST, Shakespeare, StackOverflow, etc.
"""

from hermes_unified.data.loaders import (
    load_real_federated_data,
    RealDataAdapter,
    SyntheticFEMNISTLoader,
    SyntheticShakespeareLoader
)
from hermes_unified.data.models import (
    create_model_for_dataset,
    SimpleCNNForFEMNIST,
    SimpleRNNForShakespeare
)

__all__ = [
    'load_real_federated_data',
    'RealDataAdapter',
    'SyntheticFEMNISTLoader',
    'SyntheticShakespeareLoader',
    'create_model_for_dataset',
    'SimpleCNNForFEMNIST',
    'SimpleRNNForShakespeare'
]
