"""

"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FederatedContinualCoordinator:
    """
    
    """

    def __init__(self, num_clients: int,
                 num_features: int,
                 num_classes: int = 10,
                 method: str = 'ewc'):
        self.num_clients = num_clients
        self.num_features = num_features
        self.num_classes = num_classes
        self.method = method

        self.global_weights = np.random.randn(num_features, num_classes) * 0.01
        self.global_bias = np.zeros(num_classes)

        self.clients = {}

        self.round_history = []
        self.drift_history = []
        self.forgetting_scores = {}

        self.previous_losses = {}
        self.drift_detected = False

        logger.info(f"Continual Learning Coordinator initialized")

    def create_client(self, client_id: int,
                    ewc_lambda: float = 1000.0,
                    buffer_size: int = 200):
        """"""
        if self.method == 'ewc':
            from .ewc_client import EWCClient
            client = EWCClient(client_id, self.num_features, self.num_classes, ewc_lambda)
        elif self.method == 'replay':
            from .replay_client import ReplayClient
            client = ReplayClient(client_id, self.num_features, self.num_classes, buffer_size)
        elif self.method == 'generative':
            from .replay_client import GenerativeReplayClient
            client = GenerativeReplayClient(client_id, self.num_features, self.num_classes, buffer_size=buffer_size)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.clients[client_id] = client
        return client

    def broadcast_model(self):
        """"""
        for client in self.clients.values():
            client.set_parameters({
                'weights': self.global_weights,
                'bias': self.global_bias
            })

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
            new_weights += weight * update.get('weights', np.zeros_like(self.global_weights))
            new_bias += weight * update.get('bias', np.zeros_like(self.global_bias))

        self.global_weights = new_weights
        self.global_bias = new_bias

        return {
            'weights': self.global_weights,
            'bias': self.global_bias
        }

    def detect_concept_drift(self, client_id: int,
                            old_loss: float,
                            new_loss: float,
                            threshold: float = 0.2) -> Tuple[bool, float]:
        """"""
        if self.previous_losses.get(client_id) is None:
            self.previous_losses[client_id] = new_loss
            return False, 0.0

        self.previous_losses[client_id] = new_loss

        is_drifted = new_loss > old_loss * (1 + threshold)
        drift_severity = max(0, (new_loss - old_loss) / max(old_loss, 0.01))

        if is_drifted:
            self.drift_detected = True
            self.drift_history.append({
                'round': len(self.round_history),
                'client': client_id,
                'severity': drift_severity
            })
            logger.warning(f"Concept drift detected for client {client_id}")

        return is_drifted, drift_severity

    def run_round(self, round_num: int,
                selected_clients: List[int],
                client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                num_epochs: int = 5,
                lr: float = 0.01) -> Dict[str, Any]:
        """"""
        self.round_history.append({'round': round_num, 'clients': selected_clients})

        self.broadcast_model()

        client_updates = []
        client_losses = {}
        client_drift_severities = {}

        for cid in selected_clients:
            if cid not in self.clients:
                continue

            client = self.clients[cid]
            X, y = client_data[cid]

            logits = X @ self.global_weights + self.global_bias
            probs = self._softmax(logits)
            y_onehot = np.eye(self.num_classes)[y]
            old_loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))

            train_result = client.local_train(X, y, num_epochs=num_epochs, lr=lr)

            new_logits = X @ client.weights + client.bias
            new_probs = self._softmax(new_logits)
            new_loss = -np.mean(np.sum(y_onehot * np.log(new_probs + 1e-8), axis=1))

            client_losses[cid] = new_loss

            is_drifted, drift_severity = self.detect_concept_drift(
                cid, old_loss, new_loss
            )
            client_drift_severities[cid] = drift_severity

            client_updates.append(client.get_parameters())

        self.aggregate(client_updates)

        avg_client_loss = np.mean(list(client_losses.values()))

        return {
            'round': round_num,
            'num_participants': len(selected_clients),
            'avg_loss': avg_client_loss,
            'drift_detected': len(self.drift_history) > 0,
            'drift_severities': client_drift_severities
        }

    def get_forgetting_score(self, client_id: int,
                           initial_accuracy: float,
                           current_accuracy: float) -> float:
        """"""
        forgetting = max(0, initial_accuracy - current_accuracy)

        if client_id not in self.forgetting_scores:
            self.forgetting_scores[client_id] = []

        self.forgetting_scores[client_id].append(forgetting)

        return forgetting

    def get_average_forgetting(self) -> float:
        """"""
        if not self.forgetting_scores:
            return 0.0

        latest_forgettings = []
        for scores in self.forgetting_scores.values():
            if scores:
                latest_forgettings.append(scores[-1])

        return np.mean(latest_forgettings) if latest_forgettings else 0.0

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def run_continual_learning_demo():
    """"""
    print("=" * 70)
    print("FEDERATED CONTINUAL LEARNING DEMO")
    print("=" * 70)

    np.random.seed(42)

    num_clients = 5
    num_rounds = 10
    num_features = 20
    num_classes = 5

    methods = ['ewc', 'replay']

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"Testing Method: {method.upper()}")
        print(f"{'=' * 60}")

        coordinator = FederatedContinualCoordinator(
            num_clients=num_clients,
            num_features=num_features,
            num_classes=num_classes,
            method=method
        )

        for cid in range(num_clients):
            coordinator.create_client(cid, ewc_lambda=1000, buffer_size=200)

        from .drift_simulator import DriftSimulator, DriftType

        simulator = DriftSimulator(
            num_clients=num_clients,
            num_features=num_features,
            num_classes=num_classes
        )
        simulator.configure_mixed_drift(gradual_ratio=0.5, periodic_ratio=0.3, sudden_ratio=0.2)

        round_losses = []

        for round_num in range(num_rounds):
            all_data = simulator.simulate_round(round_num)

            selected = np.random.choice(num_clients, 3, replace=False)

            result = coordinator.run_round(
                round_num=round_num,
                selected_clients=selected,
                client_data={cid: all_data[cid] for cid in selected},
                num_epochs=3,
                lr=0.1
            )

            round_losses.append(result['avg_loss'])

            if round_num % 3 == 0:
                print(f"  Round {round_num}: loss={result['avg_loss']:.4f}")

        avg_forgetting = coordinator.get_average_forgetting()
        print(f"\n  Final avg loss: {np.mean(round_losses[-3:]):.4f}")
        print(f"  Avg forgetting: {avg_forgetting:.4f}")

    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_continual_learning_demo()