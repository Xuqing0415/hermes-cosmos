"""
Federated Reinforcement Learning Module

Implements federated learning for reinforcement learning agents
with heterogeneous environments and trust region constraints.
"""

from .federated_rl import (
    PolicyNetwork,
    ValueNetwork,
    HeterogeneousEnv,
    RLClient,
    FederatedRLServer,
    run_federated_vs_centralized
)

__all__ = [
    'PolicyNetwork',
    'ValueNetwork',
    'HeterogeneousEnv',
    'RLClient',
    'FederatedRLServer',
    'run_federated_vs_centralized'
]
