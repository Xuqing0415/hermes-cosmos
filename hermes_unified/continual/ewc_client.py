"""
EWC 
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FisherInformationCalculator:
    """Fisher """

    def __init__(self, num_params: int):
        self.num_params = num_params
        self.fisher_diagonal = np.zeros(num_params)
        self.n_samples = 0

    def update(self, gradients: np.ndarray, batch_size: int):
        """ Fisher """
        grad_squared = gradients ** 2
        alpha = 0.9
        self.fisher_diagonal = alpha * self.fisher_diagonal + (1 - alpha) * grad_squared
        self.n_samples += batch_size

    def get_fisher(self) -> np.ndarray:
        """ Fisher """
        if self.n_samples > 0:
            return self.fisher_diagonal / self.n_samples
        return self.fisher_diagonal


class EWCClient:
    """
    EWC 
    """

    def __init__(self, client_id: int,
                 num_features: int,
                 num_classes: int = 10,
                 ewc_lambda: float = 1000.0):
        self.client_id = client_id
        self.num_features = num_features
        self.num_classes = num_classes
        self.ewc_lambda = ewc_lambda

        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros(num_classes)

        self.old_weights = None
        self.old_bias = None

        self.fisher_weights = FisherInformationCalculator(num_features * num_classes)
        self.fisher_bias = FisherInformationCalculator(num_classes)

        self.task_id = 0

        logger.info(f"EWC Client {client_id} initialized")

    def set_old_parameters(self, weights: np.ndarray, bias: np.ndarray):
        """"""
        self.old_weights = weights.copy()
        self.old_bias = bias.copy()

    def compute_ewc_penalty(self) -> float:
        """ EWC """
        if self.old_weights is None or self.old_bias is None:
            return 0.0

        weight_diff = self.weights - self.old_weights
        weight_penalty = np.sum(self.fisher_weights.get_fisher() * (weight_diff ** 2))

        bias_diff = self.bias - self.old_bias
        bias_penalty = np.sum(self.fisher_bias.get_fisher() * (bias_diff ** 2))

        return self.ewc_lambda * (weight_penalty + bias_penalty)

    def compute_ewc_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """ EWC """
        if self.old_weights is None or self.old_bias is None:
            return np.zeros_like(self.weights), np.zeros_like(self.bias)

        weight_diff = self.weights - self.old_weights
        weight_grad = 2 * self.ewc_lambda * self.fisher_weights.get_fisher() * weight_diff

        bias_diff = self.bias - self.old_bias
        bias_grad = 2 * self.ewc_lambda * self.fisher_bias.get_fisher() * bias_diff

        return weight_grad, bias_grad

    def local_train(self, X: np.ndarray, y: np.ndarray,
                  num_epochs: int = 5,
                  lr: float = 0.01) -> Dict[str, float]:
        """"""
        n_samples = len(y)

        for epoch in range(num_epochs):
            logits = X @ self.weights + self.bias
            probs = self._softmax(logits)

            y_onehot = np.eye(self.num_classes)[y]
            ce_loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))

            ewc_penalty = self.compute_ewc_penalty()

            total_loss = ce_loss + ewc_penalty

            grad_logits = probs - y_onehot

            grad_weights = X.T @ grad_logits / n_samples
            grad_bias = np.mean(grad_logits, axis=0)

            ewc_weight_grad, ewc_bias_grad = self.compute_ewc_gradient()

            self.weights -= lr * (grad_weights + ewc_weight_grad)
            self.bias -= lr * (grad_bias + ewc_bias_grad)

            self.fisher_weights.update(grad_weights.flatten(), n_samples)
            self.fisher_bias.update(grad_bias, n_samples)

        self.set_old_parameters(self.weights, self.bias)

        return {
            'ce_loss': ce_loss,
            'ewc_penalty': ewc_penalty,
            'total_loss': total_loss
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """"""
        logits = X @ self.weights + self.bias
        return np.argmax(logits, axis=1)

    def get_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """"""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy()
        }

    def set_parameters(self, params: Dict):
        """"""
        if 'weights' in params:
            self.weights = params['weights'].copy()
        if 'bias' in params:
            self.bias = params['bias'].copy()
        if 'task_id' in params:
            self.task_id = params['task_id']

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)