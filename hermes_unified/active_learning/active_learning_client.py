"""

"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ActiveLearningClient:
    """
    

    
    1. 
    2. 
    3. 
    """

    def __init__(self, client_id: int,
                 num_features: int,
                 num_classes: int = 10,
                 model: Any = None):
        self.client_id = client_id
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = model

        self.X_labeled = None
        self.y_labeled = None
        self.X_unlabeled = None

        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros(num_classes)

        from .uncertainty_sampling import UncertaintySampler
        self.sampler = UncertaintySampler(method='entropy')

        logger.info(f"Active Learning Client {client_id} initialized")

    def set_labeled_data(self, X: np.ndarray, y: np.ndarray):
        """"""
        self.X_labeled = X
        self.y_labeled = y

    def add_unlabeled_pool(self, X: np.ndarray):
        """"""
        self.X_unlabeled = X
        logger.info(f"Client {self.client_id}: Added {len(X)} unlabeled samples")

    def set_model_parameters(self, weights: np.ndarray, bias: np.ndarray):
        """"""
        self.weights = weights.copy()
        self.bias = bias.copy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """"""
        logits = X @ self.weights + self.bias
        return self._softmax(logits)

    def get_feature_embeddings(self, X: np.ndarray) -> np.ndarray:
        """"""
        embeddings = X @ self.weights
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        return embeddings

    def select_samples_for_labeling(self,
                                  budget: int,
                                  method: str = 'entropy',
                                  diversity_enhanced: bool = True) -> Dict[str, Any]:
        """
        
        """
        if self.X_unlabeled is None or len(self.X_unlabeled) == 0:
            logger.warning(f"Client {self.client_id}: No unlabeled samples")
            return {'selected_indices': [], 'embeddings': None}

        probs = self.predict_proba(self.X_unlabeled)

        if diversity_enhanced:
            X_selected, probs_selected, uncertainty = self.sampler.select_diverse_k(
                self.X_unlabeled, probs, budget, n_clusters=min(10, budget)
            )
        else:
            X_selected, probs_selected, uncertainty = self.sampler.select_top_k(
                self.X_unlabeled, probs, budget
            )

        embeddings = self.get_feature_embeddings(X_selected)

        selected_indices = np.arange(len(self.X_unlabeled))[:len(X_selected)]

        mask = np.ones(len(self.X_unlabeled), dtype=bool)
        mask[selected_indices] = False
        self.X_unlabeled = self.X_unlabeled[mask]

        return {
            'client_id': self.client_id,
            'selected_indices': selected_indices,
            'embeddings': embeddings,
            'uncertainty_scores': uncertainty,
            'num_selected': len(X_selected)
        }

    def add_labels(self, indices: np.ndarray, labels: np.ndarray,
                  embedding_to_label: Dict = None):
        """"""
        new_X = self.X_unlabeled[indices] if hasattr(self, 'X_unlabeled') else None

        if new_X is not None and len(new_X) > 0:
            if self.X_labeled is None:
                self.X_labeled = new_X
                self.y_labeled = labels
            else:
                self.X_labeled = np.vstack([self.X_labeled, new_X])
                self.y_labeled = np.concatenate([self.y_labeled, labels])

            logger.info(f"Client {self.client_id}: Added {len(labels)} labeled samples")

    def local_train(self, num_epochs: int = 5,
                  lr: float = 0.01) -> Dict[str, float]:
        """"""
        if self.X_labeled is None or len(self.X_labeled) < 10:
            return {'loss': 0.0, 'accuracy': 0.0}

        X = self.X_labeled
        y = self.y_labeled

        for epoch in range(num_epochs):
            logits = X @ self.weights + self.bias
            probs = self._softmax(logits)

            y_onehot = np.eye(self.num_classes)[y]
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))

            grad_logits = probs - y_onehot
            grad_weights = X.T @ grad_logits / len(y)
            grad_bias = np.mean(grad_logits, axis=0)

            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias

        predictions = np.argmax(self.predict_proba(X), axis=1)
        accuracy = np.mean(predictions == y)

        return {'loss': loss, 'accuracy': accuracy}

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """"""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy()
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)