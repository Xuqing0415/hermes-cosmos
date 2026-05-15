"""
Experiments Module

Real data benchmarks and experiment runners.
"""

from hermes_unified.experiments.real_data_benchmark import (
    run_real_data_benchmark,
    compare_algorithms,
    RealDataFederatedClient,
    RealDataFederatedServer
)

__all__ = [
    'run_real_data_benchmark',
    'compare_algorithms',
    'RealDataFederatedClient',
    'RealDataFederatedServer'
]
