"""
Meta-FL Controller Module

Automatic algorithm selection for federated learning tasks.
"""

from .feature_extractor import TaskFeatureExtractor, compute_non_iid_metrics, generate_synthetic_task
from .performance_predictor import PerformancePredictor
from .recommender import AlgorithmRecommender

__all__ = [
    'TaskFeatureExtractor',
    'compute_non_iid_metrics',
    'generate_synthetic_task',
    'PerformancePredictor',
    'AlgorithmRecommender'
]
