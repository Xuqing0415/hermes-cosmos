"""
记忆重放客户端实现
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ReplayClient:
    """
    记忆重放客户端
    """

    def __init__(self, client_id: int,
                 num_features: int,
                 num_classes: int = 10,
                 buffer_size: int = 200,
                 samples_per_class: int = 20):
        self.client_id = client_id
        self.num_features = num_features
        self.num_classes = num_classes
        self.buffer_size = buffer_size
        self.samples_per_class = samples_per_class

        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros(num_classes)

        self.memory_buffer_X = []
        self.memory_buffer_y = []

        logger.info(f"Replay Client {client_id} initialized")

    def update_memory(self, X: np.ndarray, y: np.ndarray):
        """更新记忆缓冲区"""
        for c in range(self.num_classes):
            mask = y == c
            X_c = X[mask]
            y_c = y[mask]

            if len(X_c) == 0:
                continue

            existing_X, existing_y = self._get_class_from_buffer(c)

            if len(existing_X) >= self.samples_per_class:
                continue

            needed = self.samples_per_class - len(existing_X)

            if needed > 0 and len(X_c) > 0:
                indices = np.random.choice(len(X_c), min(needed, len(X_c)), replace=False)
                selected_X = X_c[indices]
                selected_y = y_c[indices]

                self.memory_buffer_X.extend(selected_X)
                self.memory_buffer_y.extend(selected_y)

        self._trim_buffer()

    def _get_class_from_buffer(self, target_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """从缓冲区获取特定类别的样本"""
        X_selected = []
        y_selected = []

        for i, y in enumerate(self.memory_buffer_y):
            if y == target_class:
                X_selected.append(self.memory_buffer_X[i])
                y_selected.append(y)

        if X_selected:
            return np.array(X_selected), np.array(y_selected)
        return np.array([]), np.array([])

    def _trim_buffer(self):
        """淘汰多余样本"""
        total_size = len(self.memory_buffer_X)

        if total_size > self.buffer_size:
            remove_count = total_size - self.buffer_size
            remove_indices = np.random.choice(total_size, remove_count, replace=False)

            new_X = [x for i, x in enumerate(self.memory_buffer_X)
                    if i not in remove_indices]
            new_y = [y for i, y in enumerate(self.memory_buffer_y)
                    if i not in remove_indices]

            self.memory_buffer_X = new_X
            self.memory_buffer_y = new_y

    def get_replay_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取重放样本"""
        if not self.memory_buffer_X:
            return np.array([]), np.array([])

        return np.array(self.memory_buffer_X), np.array(self.memory_buffer_y)

    def local_train(self, X: np.ndarray, y: np.ndarray,
                  num_epochs: int = 5,
                  lr: float = 0.01,
                  replay_ratio: float = 0.3) -> Dict[str, float]:
        """本地训练"""
        self.update_memory(X, y)

        replay_X, replay_y = self.get_replay_samples()

        n_current = len(X)
        n_replay = len(replay_X)

        if n_replay == 0:
            return self._train_batch(X, y, num_epochs, lr)

        if replay_ratio > 0 and n_current > 0:
            n_replay_use = int(n_current * replay_ratio / (1 - replay_ratio))
            n_replay_use = min(n_replay_use, n_replay)

            indices = np.random.choice(n_replay, n_replay_use, replace=False)
            mixed_X = np.vstack([X, replay_X[indices]])
            mixed_y = np.concatenate([y, replay_y[indices]])
        else:
            mixed_X = X
            mixed_y = y

        return self._train_batch(mixed_X, mixed_y, num_epochs, lr)

    def _train_batch(self, X: np.ndarray, y: np.ndarray,
                   num_epochs: int, lr: float) -> Dict[str, float]:
        """标准批量训练"""
        n_samples = len(y)

        for epoch in range(num_epochs):
            logits = X @ self.weights + self.bias
            probs = self._softmax(logits)

            y_onehot = np.eye(self.num_classes)[y]
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))

            grad_logits = probs - y_onehot
            grad_weights = X.T @ grad_logits / n_samples
            grad_bias = np.mean(grad_logits, axis=0)

            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias

        return {'loss': loss}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        logits = X @ self.weights + self.bias
        return np.argmax(logits, axis=1)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """获取模型参数"""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy()
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class GenerativeReplayClient(ReplayClient):
    """生成式重放客户端"""

    def __init__(self, client_id: int,
                 num_features: int,
                 num_classes: int = 10,
                 latent_dim: int = 20,
                 buffer_size: int = 200):
        super().__init__(client_id, num_features, num_classes, buffer_size)

        self.latent_dim = latent_dim

        self.generator_weights = np.random.randn(latent_dim, num_features) * 0.01
        self.generator_bias = np.zeros(num_features)

        logger.info(f"Generative Replay Client {client_id} initialized")

    def train_generator(self, X: np.ndarray, y: np.ndarray,
                      num_epochs: int = 10, lr: float = 0.01):
        """训练生成器"""
        n_samples = len(X)

        for epoch in range(num_epochs):
            noise = np.random.randn(n_samples, self.latent_dim)
            generated = noise @ self.generator_weights + self.generator_bias

            reconstruction_loss = np.mean((X - generated) ** 2)

            grad_weights = -2 * (X - generated).T @ noise / n_samples
            grad_bias = -2 * np.mean(X - generated, axis=0)

            self.generator_weights -= lr * grad_weights
            self.generator_bias -= lr * grad_bias

    def generate_pseudo_samples(self, target_class: int,
                              num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """为指定类别生成伪样本"""
        noise = np.random.randn(num_samples, self.latent_dim)

        class_offset = np.zeros(self.latent_dim)
        class_offset[target_class % self.latent_dim] = 1.0

        noise_with_class = noise + 0.5 * class_offset

        generated = noise_with_class @ self.generator_weights + self.generator_bias

        labels = np.full(num_samples, target_class)

        return generated, labels

    def get_replay_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取重放样本"""
        all_X = []
        all_y = []

        for c in range(self.num_classes):
            gen_X, gen_y = self.generate_pseudo_samples(c, num_samples=10)
            all_X.append(gen_X)
            all_y.append(gen_y)

        return np.vstack(all_X), np.concatenate(all_y)