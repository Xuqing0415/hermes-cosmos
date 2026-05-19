"""
Federated Causal Representation Learning Module

Implements federated learning for causal representations, including:
- CausalVAE for disentangled causal representation learning
- Distributed causal graph discovery
- Invariance constraints across environments
- Counterfactual reasoning
"""

from .causal_vae_client import (
    CausalVAEClient,
    CausalEncoder,
    CausalDecoder,
    StructuralEquationModel
)

from .graph_aggregator import (
    CausalGraphAggregator,
    PrivacyPreservingGraphAggregator,
    GraphKernel
)

from .invariance_constraint import (
    InvarianceConstraint,
    MMDInvariance,
    AdversarialInvariance
)

from .counterfactual import (
    CounterfactualReasoner,
    InterventionModel
)

from .fed_causal_coordinator import (
    FedCausalCoordinator,
    run_fed_causal_demo
)

__all__ = [
    'CausalVAEClient',
    'CausalEncoder',
    'CausalDecoder',
    'StructuralEquationModel',
    'CausalGraphAggregator',
    'PrivacyPreservingGraphAggregator',
    'GraphKernel',
    'InvarianceConstraint',
    'MMDInvariance',
    'AdversarialInvariance',
    'CounterfactualReasoner',
    'InterventionModel',
    'FedCausalCoordinator',
    'run_fed_causal_demo'
]