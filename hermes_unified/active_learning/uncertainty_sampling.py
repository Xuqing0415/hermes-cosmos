"""
不确定性采样策略
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UncertaintySampler:
    """
    不确定性采样器

    支持多种不确定性度量方法
    """

    def __init__(self, method: str = 'entropy'):
        """
        Args:
            method: 'entropy', 'margin', 'least_confidence', 'coreset'
        """
        self.method = method

    def compute_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """
        计算每个样本的不确定性分数

        Args:
            probabilities: softmax 概率 (n_samples, n_classes)

        Returns:
            不确定性分数 (n_samples,)
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
        """熵不确定性: -sum(p * log(p))"""
        eps = 1e-10
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        return entropy

    def _margin(self, probs: np.ndarray) -> np.ndarray:
        """边际不确定性: p1 - p2 (最大与次大概率的差)"""
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        return 1 - margins

    def _least_confidence(self, probs: np.ndarray) -> np.ndarray:
        """最小置信度: 1 - max(p)"""
        max_probs = np.max(probs, axis=1)
        return 1 - max_probs

    def select_top_k(self, X_pool: np.ndarray,
                   probabilities: np.ndarray,
                   k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        选择 Top-K 最不确定的样本

        Args:
            X_pool: 未标注样本池
            probabilities: 模型预测概率
            k: 选择数量

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
        多样性增强选择：从每个簇中选择最不确定的样本
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
    核心集评分：选择能覆盖特征空间的样本
    """
    X_labeled_norm = X_labeled / (np.linalg.norm(X_labeled, axis=1, keepdims=True) + 1e-10)
    X_pool_norm = X_pool / (np.linalg.norm(X_pool, axis=1, keepdims=True) + 1e-10)

    distances = np.zeros(len(X_pool))

    for i in range(len(X_pool)):
        dists = np.sum((X_pool_norm[i] - X_labeled_norm) ** 2, axis=1)
        distances[i] = np.min(dists)

    coreset_scores = distances

    return coreset_scores