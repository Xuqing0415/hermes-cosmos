"""
Lightweight Edge Client for Federated Learning

A minimal federated learning client that runs on resource-constrained devices
(Raspberry Pi, old phones, laptops) with only basic dependencies (requests, numpy).
"""

import requests
import numpy as np
import time
import json
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeClient:
    """
    Lightweight federated learning client for edge devices.

    Supports:
    - Pulling global model from server
    - Local training (simulated or real)
    - Uploading model updates
    - Auto-reconnection on failure
    - Local data caching
    """

    def __init__(self, server_url: str, device_id: str, local_data: List[Tuple],
                 model_size: int = 100, batch_size: int = 32, lr: float = 0.01):
        """
        Initialize edge client.

        Args:
            server_url: URL of the federated learning server
            device_id: Unique identifier for this device
            local_data: List of (x, y) tuples for local training
            model_size: Size of the model parameter vector
            batch_size: Training batch size
            lr: Learning rate
        """
        self.server_url = server_url.rstrip('/')
        self.device_id = device_id
        self.local_data = local_data
        self.model_size = model_size
        self.batch_size = batch_size
        self.lr = lr

        self.model = np.zeros(model_size)
        self.round_number = 0
        self.training_stats = {
            'local_epochs': 0,
            'loss': 0.0,
            'samples_trained': 0
        }

        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def pull_model(self) -> bool:
        """
        Pull the latest global model from the server.

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.server_url}/get_model",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            self.model = np.array(data['weights'])

            if 'round' in data:
                self.round_number = data['round']

            logger.info(f"Device {self.device_id}: Pulled model (round {self.round_number})")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Device {self.device_id}: Failed to pull model - {e}")
            return False

    def push_update(self, update: np.ndarray) -> bool:
        """
        Push local model update to the server.

        Args:
            update: Model update (local_model - global_model)

        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {
                'device_id': self.device_id,
                'round': self.round_number,
                'update': update.tolist(),
                'stats': self.training_stats
            }

            response = self.session.post(
                f"{self.server_url}/submit_update",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            logger.info(f"Device {self.device_id}: Pushed update successfully")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Device {self.device_id}: Failed to push update - {e}")
            return False

    def train_local(self, epochs: int = 1) -> np.ndarray:
        """
        Perform local training on the device's data.

        Args:
            epochs: Number of local training epochs

        Returns:
            Trained model parameters
        """
        X, y = self._prepare_data()

        self.training_stats['local_epochs'] = epochs
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            indices = np.random.permutation(len(y))

            for i in range(0, len(y), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                loss = self._train_step(X_batch, y_batch)
                total_loss += loss
                num_batches += 1

        self.training_stats['loss'] = total_loss / max(num_batches, 1)
        self.training_stats['samples_trained'] = len(y)

        logger.info(f"Device {self.device_id}: Trained for {epochs} epochs, loss={self.training_stats['loss']:.4f}")

        return self.model.copy()

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare local data for training."""
        if not self.local_data:
            X = np.random.randn(100, self.model_size) * 0.1
            y = np.random.randint(0, 10, size=100)
        else:
            X = np.array([d[0] for d in self.local_data])
            y = np.array([d[1] for d in self.local_data])

        return X, y

    def _train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform one training step (gradient descent on linear model).

        Args:
            X: Input features
            y: Labels

        Returns:
            Loss value
        """
        predictions = X @ self.model
        errors = predictions - y

        gradient = X.T @ errors / len(y)
        self.model -= self.lr * gradient

        loss = np.mean(errors ** 2)
        return loss

    def run(self, poll_interval: int = 30, max_rounds: Optional[int] = None):
        """
        Run the federated learning client loop.

        Args:
            poll_interval: Seconds between each participation round
            max_rounds: Maximum number of rounds (None for infinite)
        """
        self.running = True
        consecutive_failures = 0
        max_failures = 5

        logger.info(f"Device {self.device_id}: Starting federated learning client")

        while self.running:
            try:
                if not self.pull_model():
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.warning(f"Device {self.device_id}: Too many failures, waiting longer...")
                        time.sleep(poll_interval * 3)
                    continue

                consecutive_failures = 0

                local_model = self.train_local(epochs=1)

                update = local_model - self.model

                if self.push_update(update):
                    self.round_number += 1

                if max_rounds and self.round_number >= max_rounds:
                    logger.info(f"Device {self.device_id}: Completed {max_rounds} rounds")
                    break

                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Device {self.device_id}: Unexpected error - {e}")
                time.sleep(10)

        logger.info(f"Device {self.device_id}: Client stopped")

    def stop(self):
        """Stop the client loop."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def start(self, poll_interval: int = 30, max_rounds: Optional[int] = None):
        """Start the client in a background thread."""
        self.thread = threading.Thread(
            target=self.run,
            args=(poll_interval, max_rounds),
            daemon=True
        )
        self.thread.start()

    def get_status(self) -> Dict[str, Any]:
        """Get current client status."""
        return {
            'device_id': self.device_id,
            'round': self.round_number,
            'model_norm': float(np.linalg.norm(self.model)),
            'training_stats': self.training_stats,
            'running': self.running
        }


class TemperaturePredictionClient(EdgeClient):
    """
    Specialized edge client for temperature prediction task.

    Uses polynomial regression for temperature forecasting.
    """

    def __init__(self, server_url: str, device_id: str, temperature_data: List[float],
                 time_features: List[List[float]], model_size: int = 50):
        """
        Initialize temperature prediction client.

        Args:
            server_url: Server URL
            device_id: Device identifier
            temperature_data: List of temperature readings
            time_features: List of [hour, day_of_week, month] features
            model_size: Model parameter size
        """
        local_data = list(zip(time_features, temperature_data))
        super().__init__(server_url, device_id, local_data, model_size)

    def _train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Override training step for temperature prediction."""
        predictions = X @ self.model
        errors = predictions - y

        gradient = X.T @ errors / len(y)
        self.model -= self.lr * gradient

        mse = np.mean(errors ** 2)
        return mse


class ImageClassificationClient(EdgeClient):
    """
    Specialized edge client for image classification task.

    Uses a simple linear classifier with extracted features.
    """

    def __init__(self, server_url: str, device_id: str, features: np.ndarray,
                 labels: np.ndarray, model_size: int = 784):
        """
        Initialize image classification client.

        Args:
            server_url: Server URL
            device_id: Device identifier
            features: Flattened image features (N x feature_dim)
            labels: Class labels (N,)
            model_size: Size of feature vector
        """
        local_data = list(zip(features, labels))
        super().__init__(server_url, device_id, local_data, model_size)

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract simple features from images (edge detection + histogram).

        Args:
            images: Raw image arrays

        Returns:
            Feature vectors
        """
        features = []

        for img in images:
            img = img.flatten()

            edges = np.abs(np.diff(img))
            edge_sum = np.sum(edges)

            hist, _ = np.histogram(img, bins=10)

            feature = np.concatenate([img[:50], [edge_sum], hist / (np.sum(hist) + 1e-6)])
            features.append(feature[:self.model_size])

        return np.array(features)


def create_temperature_data(num_samples: int = 100, seed: int = None) -> Tuple[List[float], List[List[float]]]:
    """
    Generate synthetic temperature data for testing.

    Args:
        num_samples: Number of data points
        seed: Random seed

    Returns:
        Tuple of (temperatures, time_features)
    """
    if seed:
        np.random.seed(seed)

    temperatures = []
    time_features = []

    base_temp = 20.0

    for i in range(num_samples):
        hour = (i % 24)
        day = (i // 24) % 7
        month = (i // 720) % 12 + 1

        temp = base_temp + 5 * np.sin(2 * np.pi * hour / 24) + \
               3 * np.sin(2 * np.pi * day / 7) + \
               np.random.randn() * 0.5

        temperatures.append(temp)
        time_features.append([hour / 24.0, day / 7.0, month / 12.0])

    return temperatures, time_features


if __name__ == "__main__":
    print("=== Testing Lightweight Edge Client ===")

    temps, features = create_temperature_data(100)

    client = TemperaturePredictionClient(
        server_url="http://localhost:8000",
        device_id="test_device_1",
        temperature_data=temps,
        time_features=features
    )

    print(f"Client status: {client.get_status()}")

    client.start(poll_interval=5, max_rounds=2)

    time.sleep(3)

    print(f"Client status after training: {client.get_status()}")

    client.stop()

    print("Test complete!")
