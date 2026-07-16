"""

"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PseudoLabelFilter:
    """"""

    def __init__(self, initial_threshold: float = 0.9,
                 final_threshold: float = 0.95,
                 warmup_rounds: int = 5):
        self.threshold = initial_threshold
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.warmup_rounds = warmup_rounds
        self.current_round = 0

    def update_threshold(self, round_num: int):
        """"""
        if round_num < self.warmup_rounds:
            self.threshold = self.initial_threshold
        else:
            progress = (round_num - self.warmup_rounds) / (100 - self.warmup_rounds)
            self.threshold = self.initial_threshold + progress * (self.final_threshold - self.initial_threshold)

    def filter_pseudo_labels(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        max_probs = np.max(probs, axis=1)
        pseudo_labels = np.argmax(probs, axis=1)

        mask = max_probs >= self.threshold

        return mask, pseudo_labels


class ConsistencyRegularizer:
    """"""

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def compute_consistency_loss(self,
                               weak_probs: np.ndarray,
                               strong_probs: np.ndarray) -> float:
        """ (KL )"""
        eps = 1e-10

        kl_div = np.sum(weak_probs * np.log(weak_probs / (strong_probs + eps) + eps), axis=1)

        return self.strength * np.mean(kl_div)

    def augment_weak(self, X: np.ndarray) -> np.ndarray:
        """"""
        X_aug = X.copy()
        flip_mask = np.random.rand(len(X)) > 0.5
        X_aug[flip_mask] = -X_aug[flip_mask]
        return X_aug

    def augment_strong(self, X: np.ndarray) -> np.ndarray:
        """ + """
        X_aug = X.copy()

        flip_mask = np.random.rand(len(X)) > 0.5
        X_aug[flip_mask] = -X_aug[flip_mask]

        noise = np.random.randn(*X_aug.shape) * 0.1
        X_aug = X_aug + noise

        return X_aug


class SemiSupervisedClient:
    """
    

     MixMatch/FixMatch 
    """

    def __init__(self, client_id: int,
                 num_features: int,
                 num_classes: int = 10,
                 unlabeled_weight: float = 1.0,
                 consistency_strength: float = 1.0):
        self.client_id = client_id
        self.num_features = num_features
        self.num_classes = num_classes
        self.unlabeled_weight = unlabeled_weight

        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros(num_classes)

        self.pseudo_filter = PseudoLabelFilter()
        self.consistency_reg = ConsistencyRegularizer(strength=consistency_strength)

        self.X_labeled = None
        self.y_labeled = None
        self.X_unlabeled = None

        logger.info(f"Semi-supervised Client {client_id} initialized")

    def set_data(self, X_labeled: np.ndarray, y_labeled: np.ndarray,
                X_unlabeled: np.ndarray):
        """"""
        self.X_labeled = X_labeled
        self.y_labeled = y_labeled
        self.X_unlabeled = X_unlabeled

        logger.info(f"Client {self.client_id}: {len(X_labeled)} labeled, {len(X_unlabeled)} unlabeled")

    def set_model_parameters(self, weights: np.ndarray, bias: np.ndarray):
        """"""
        self.weights = weights.copy()
        self.bias = bias.copy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """"""
        logits = X @ self.weights + self.bias
        return self._softmax(logits)

    def local_train(self,
                  num_epochs: int = 5,
                  lr: float = 0.01,
                  round_num: int = 0) -> Dict[str, float]:
        """"""
        self.pseudo_filter.update_threshold(round_num)

        if self.X_labeled is not None and len(self.X_labeled) > 0:
            sup_loss, sup_acc = self._supervised_loss(self.X_labeled, self.y_labeled, lr, num_epochs)
        else:
            sup_loss, sup_acc = 0.0, 0.0

        unsup_loss = 0.0
        if self.X_unlabeled is not None and len(self.X_unlabeled) > 0:
            unsup_loss = self._unsupervised_loss(round_num)

        total_loss = sup_loss + self.unlabeled_weight * unsup_loss

        return {
            'supervised_loss': sup_loss,
            'unsupervised_loss': unsup_loss,
            'total_loss': total_loss,
            'supervised_accuracy': sup_acc
        }

    def _supervised_loss(self, X: np.ndarray, y: np.ndarray,
                       lr: float, num_epochs: int) -> Tuple[float, float]:
        """"""
        for epoch in range(num_epochs):
            probs = self.predict_proba(X)

            y_onehot = np.eye(self.num_classes)[y]
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))

            grad_logits = probs - y_onehot
            grad_weights = X.T @ grad_logits / len(y)
            grad_bias = np.mean(grad_logits, axis=0)

            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias

        predictions = np.argmax(self.predict_proba(X), axis=1)
        accuracy = np.mean(predictions == y)

        return loss, accuracy

    def _unsupervised_loss(self, round_num: int) -> float:
        """"""
        X_unlabeled = self.X_unlabeled

        probs = self.predict_proba(X_unlabeled)
        mask, pseudo_labels = self.pseudo_filter.filter_pseudo_labels(probs)

        if np.sum(mask) < 10:
            return 0.0

        X_confident = X_unlabeled[mask]
        pseudo_y = pseudo_labels[mask]

        X_weak = self.consistency_reg.augment_weak(X_confident)
        X_strong = self.consistency_reg.augment_strong(X_confident)

        probs_weak = self.predict_proba(X_weak)
        probs_strong = self.predict_proba(X_strong)

        consistency_loss = self.consistency_reg.compute_consistency_loss(probs_weak, probs_strong)

        y_onehot = np.eye(self.num_classes)[pseudo_y]
        pseudo_loss = -np.mean(np.sum(y_onehot * np.log(probs_strong + 1e-8), axis=1))

        return consistency_loss + 0.5 * pseudo_loss

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """"""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy()
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


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


def run_semi_supervised_demo():
    """"""
    print("=" * 70)
    print("FEDERATED SEMI-SUPERVISED LEARNING DEMO")
    print("=" * 70)

    np.random.seed(42)

    num_clients = 3
    num_features = 20
    num_classes = 5
    num_rounds = 10

    from .semi_supervised_server import SemiSupervisedServer
    server = SemiSupervisedServer(num_features, num_classes)

    clients = []
    for cid in range(num_clients):
        client = SemiSupervisedClient(cid, num_features, num_classes)

        n_labeled = 30
        X_labeled = np.random.randn(n_labeled, num_features)
        y_labeled = np.random.randint(0, num_classes, n_labeled)

        n_unlabeled = 300
        X_unlabeled = np.random.randn(n_unlabeled, num_features)

        client.set_data(X_labeled, y_labeled, X_unlabeled)
        clients.append(client)

    print("\nTraining Rounds:")

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        global_model = server.get_global_model()
        for client in clients:
            client.set_model_parameters(global_model['weights'], global_model['bias'])

        total_sup_loss = 0
        total_unsup_loss = 0
        total_acc = 0

        for client in clients:
            result = client.local_train(round_num=round_num)
            total_sup_loss += result['supervised_loss']
            total_unsup_loss += result['unsupervised_loss']
            total_acc += result['supervised_accuracy']

        client_updates = [client.get_parameters() for client in clients]
        server.aggregate(client_updates)

        avg_sup = total_sup_loss / len(clients)
        avg_unsup = total_unsup_loss / len(clients)
        avg_acc = total_acc / len(clients)

        print(f"  Avg supervised loss: {avg_sup:.4f}")
        print(f"  Avg unsupervised loss: {avg_unsup:.4f}")
        print(f"  Avg supervised accuracy: {avg_acc:.4f}")

    print("\n" + "=" * 70)
    print("SEMI-SUPERVISED LEARNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_semi_supervised_demo()