"""
Federated Online Learning Module

Implements federated learning for streaming data with concept drift handling,
including online gradient descent, drift detection, and meta-learning.
"""

from .online_client import (
    OnlineClient,
    StreamingGradientAccumulator,
    DriftAlert
)

from .online_server import (
    OnlineServer,
    AsyncGradientAggregator,
    OnlineLearningRateScheduler
)

from .drift_detector import (
    DriftDetector,
    CUSUMDetector,
    ADWINDetector,
    PageHinkleyDetector
)

from .fomaml import (
    FOMAMLClient,
    FOMAMLServer,
    FederatedOnlineMetaLearner
)

from .fed_online_fl import (
    FederatedOnlineLearning,
    run_online_demo
)

__all__ = [
    'OnlineClient',
    'StreamingGradientAccumulator',
    'DriftAlert',
    'OnlineServer',
    'AsyncGradientAggregator',
    'OnlineLearningRateScheduler',
    'DriftDetector',
    'CUSUMDetector',
    'ADWINDetector',
    'PageHinkleyDetector',
    'FOMAMLClient',
    'FOMAMLServer',
    'FederatedOnlineMetaLearner',
    'FederatedOnlineLearning',
    'run_online_demo'
]