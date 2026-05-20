"""
Federated Linear Mode Connectivity Module

Implements tools for analyzing linear mode connectivity in federated learning,
including interpolation evaluation, connectivity metrics, and visualization.
"""

from .connectivity_metrics import (
    LinearConnectivityMetrics,
    InterpolationLossEvaluator,
    ConnectivityScore,
    LossLandscapeAnalyzer
)

from .federated_lmc import (
    FederatedLMC,
    LMCCoordinator,
    ClientModelSnapshot
)

from .visualization import (
    plot_interpolation_path,
    plot_connectivity_evolution,
    plot_loss_landscape
)

__all__ = [
    'LinearConnectivityMetrics',
    'InterpolationLossEvaluator',
    'ConnectivityScore',
    'LossLandscapeAnalyzer',
    'FederatedLMC',
    'LMCCoordinator',
    'ClientModelSnapshot',
    'plot_interpolation_path',
    'plot_connectivity_evolution',
    'plot_loss_landscape'
]