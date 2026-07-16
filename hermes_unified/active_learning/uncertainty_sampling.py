"""

"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UncertaintySampler:
    """
    

    
    """

    def __init__(self, method: str = 'entropy'):
        """
        Args:
            method: 'entropy', 'margin', 'least_confidence', 'coreset'
        """
        self.method = method

    def compute_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        

        Args:
            probabilities: softmax  (n_samples, n_classes)

        Returns:
             (n_samples,)
        """
        if self.method == 'entropy':
            return self._entropy(probabilities)
        elif self.method == 'margin':
            return self._margin(probabilities)
        elif self.method == 'least_confidence':
            return self._least_confidence(probabilities)
        else:
            return self._entropy(probabilities)

    def _entropy(self, probs: np.ndarray) -> np.ndarray:
        """: -sum(p * log(p))"""
        eps = 1e-10
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        return entropy

    def _margin(self, probs: np.ndarray) -> np.ndarray:
        """: p1 - p2 ()"""
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        return 1 - margins

    def _least_confidence(self, probs: np.ndarray) -> np.ndarray:
        """: 1 - max(p)"""
        max_probs = np.max(probs, axis=1)
        return 1 - max_probs

    def select_top_k(self, X_pool: np.ndarray,
                   probabilities: np.ndarray,
                   k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
         Top-K 

        Args:
            X_pool: 
            probabilities: 
            k: 

        Returns:
            (X_selected, probs_selected, uncertainty_scores)
        """
        uncertainty = self.compute_uncertainty(probabilities)

        top_k_indices = np.argsort(uncertainty)[-k:]

        X_selected = X_pool[top_k_indices]
        probs_selected = probabilities[top_k_indices]
        uncertainty_selected = uncertainty[top_k_indices]

        return X_selected, probs_selected, uncertainty_selected

    def select_diverse_k(self, X_pool: np.ndarray,
                        probabilities: np.ndarray,
                        k: int,
                        n_clusters: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        
        """
        uncertainty = self.compute_uncertainty(probabilities)

        X_normalized = X_pool / (np.linalg.norm(X_pool, axis=1, keepdims=True) + 1e-10)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(X_pool)), random_state=42)
        cluster_labels = kmeans.fit_predict(X_normalized)

        selected_indices = []

        for cluster_id in range(kmeans.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_uncertainty = uncertainty[cluster_indices]
            n_select = max(1, k // n_clusters)

            top_in_cluster = cluster_indices[np.argsort(cluster_uncertainty)[-n_select:]]
            selected_indices.extend(top_in_cluster)

        selected_indices = selected_indices[:k]

        X_selected = X_pool[selected_indices]
        probs_selected = probabilities[selected_indices]
        uncertainty_selected = uncertainty[selected_indices]

        return X_selected, probs_selected, uncertainty_selected


def compute_coreset_scores(X_pool: np.ndarray,
                         X_labeled: np.ndarray,
                         k: int) -> np.ndarray:
    """
    
    """
    X_labeled_norm = X_labeled / (np.linalg.norm(X_labeled, axis=1, keepdims=True) + 1e-10)
    X_pool_norm = X_pool / (np.linalg.norm(X_pool, axis=1, keepdims=True) + 1e-10)

    distances = np.zeros(len(X_pool))

    for i in range(len(X_pool)):
        dists = np.sum((X_pool_norm[i] - X_labeled_norm) ** 2, axis=1)
        distances[i] = np.min(dists)

    coreset_scores = distances

    return coreset_scores