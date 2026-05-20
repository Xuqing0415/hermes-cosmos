"""
Federated Neural Architecture Search (FedNAS) Module

Implements federated neural architecture search with heterogeneous client constraints,
including supernet training, local architecture search, and subnet extraction.
"""

from .supernet import (
    SuperNet,
    MixedOp,
    SearchSpace,
    DartsSearchSpace
)

from .subnet_extractor import (
    SubnetExtractor,
    ResourceEvaluator,
    ConstraintChecker
)

from .local_search import (
    LocalSearch,
    EvolutionarySearch,
    BayesianSearch,
    DifferentiableSearch
)

from .fed_nas_fl import (
    FederatedNAS,
    run_fednas_demo
)

__all__ = [
    'SuperNet',
    'MixedOp',
    'SearchSpace',
    'DartsSearchSpace',
    'SubnetExtractor',
    'ResourceEvaluator',
    'ConstraintChecker',
    'LocalSearch',
    'EvolutionarySearch',
    'BayesianSearch',
    'DifferentiableSearch',
    'FederatedNAS',
    'run_fednas_demo'
]