"""

"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SemiSupervisedServer:
    """"""

    def __init__(self, num_features: int, num_classes: int = 10):
        self.num_features = num_features
        self.num_classes = num_classes

        self.global_weights = np.random.randn(num_features, num_classes) * 0.01
        self.global_bias = np.zeros(num_classes)

    def aggregate(self, client_updates: List[Dict],
                client_weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """"""
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)

        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        new_weights = np.zeros_like(self.global_weights)
        new_bias = np.zeros_like(self.global_bias)

        for update, weight in zip(client_updates, normalized_weights):
            new_weights += weight * update['weights']
            new_bias += weight * update['bias']

        self.global_weights = new_weights
        self.global_bias = new_bias

        return {
            'weights': self.global_weights.copy(),
            'bias': self.global_bias.copy()
        }

    def get_global_model(self) -> Dict[str, np.ndarray]:
        return {
            'weights': self.global_weights,
            'bias': self.global_bias
        }