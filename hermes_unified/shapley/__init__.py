"""
Federated Shapley Value Module

Implements fair contribution allocation using Shapley values in federated learning,
including Monte Carlo estimation and incentive mechanisms.
"""

from .shapley_estimator import (
    ShapleyEstimator,
    MonteCarloShapleyEstimator,
    GroupedShapleyEstimator,
    ShapleyValue
)

from .incentive_mechanism import (
    IncentiveMechanism,
    TokenDistributor,
    ReputationSystem
)

from .fed_shapley_fl import (
    FederatedShapleyLearning,
    run_shapley_demo
)

__all__ = [
    'ShapleyEstimator',
    'MonteCarloShapleyEstimator',
    'GroupedShapleyEstimator',
    'ShapleyValue',
    'IncentiveMechanism',
    'TokenDistributor',
    'ReputationSystem',
    'FederatedShapleyLearning',
    'run_shapley_demo'
]