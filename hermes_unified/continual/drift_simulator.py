"""
数据漂移模拟器
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DriftType:
    """漂移类型枚举"""
    GRADUAL = "gradual"
    PERIODIC = "periodic"
    SUDDEN = "sudden"
    INCREMENTAL = "incremental"


class DataBuffer:
    """固定大小的数据缓冲区"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.X_buffer = []
        self.y_buffer = []
        self.timestamps = []

    def add(self, X: np.ndarray, y: np.ndarray, timestamp: int):
        """添加数据"""
        for i in range(len(y)):
            self.X_buffer.append(X[i])
            self.y_buffer.append(y[i])
            self.timestamps.append(timestamp)

            if len(self.X_buffer) > self.max_size:
                self.X_buffer.pop(0)
                self.y_buffer.pop(0)
                self.timestamps.pop(0)

    def get_recent(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取最近的 n 个样本"""
        recent_X = np.array(self.X_buffer[-n:])
        recent_y = np.array(self.y_buffer[-n:])
        return recent_X, recent_y

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取所有缓冲数据"""
        return np.array(self.X_buffer), np.array(self.y_buffer)

    def size(self) -> int:
        """当前缓冲区大小"""
        return len(self.X_buffer)


class DriftSimulator:
    """
    数据漂移模拟器
    """

    def __init__(self, num_clients: int = 10,
                 num_features: int = 20,
                 num_classes: int = 10,
                 seed: int = 42):
        self.num_clients = num_clients
        self.num_features = num_features
        self.num_classes = num_classes
        np.random.seed(seed)

        self.client_distributions = self._init_distributions()
        self.current_round = 0

    def _init_distributions(self) -> Dict[int, Dict[str, np.ndarray]]:
        """初始化客户端分布参数"""
        distributions = {}

        for cid in range(self.num_clients):
            distributions[cid] = {
                'means': np.random.randn(self.num_classes, self.num_features) * 2,
                'covs': [np.eye(self.num_features) * 0.5 for _ in range(self.num_classes)],
                'class_weights': np.random.dirichlet(np.ones(self.num_classes)),
                'drift_speed': np.random.rand(self.num_features) * 0.1,
                'base_offset': np.random.randn(self.num_features) * 5,
                'drift_type': 'none',
                'drift_params': {}
            }

        return distributions

    def configure_client_drift(self, client_id: int, drift_type: str, **kwargs):
        """配置单个客户端的漂移模式"""
        if client_id not in self.client_distributions:
            return

        self.client_distributions[client_id]['drift_type'] = drift_type
        self.client_distributions[client_id]['drift_params'] = kwargs

    def configure_mixed_drift(self, gradual_ratio: float = 0.5,
                            periodic_ratio: float = 0.3,
                            sudden_ratio: float = 0.2):
        """配置混合漂移模式"""
        assert np.isclose(gradual_ratio + periodic_ratio + sudden_ratio, 1.0)

        drift_types = []
        drift_types.extend(['gradual'] * int(self.num_clients * gradual_ratio))
        drift_types.extend(['periodic'] * int(self.num_clients * periodic_ratio))
        drift_types.extend(['sudden'] * int(self.num_clients * sudden_ratio))

        remaining = self.num_clients - len(drift_types)
        drift_types.extend(['gradual'] * remaining)

        np.random.shuffle(drift_types)

        for cid, drift_type in enumerate(drift_types):
            params = {}
            if drift_type == 'periodic':
                params = {'period': np.random.randint(8, 15), 'amplitude': np.random.uniform(0.5, 1.5)}
            elif drift_type == 'sudden':
                params = {'drift_round': np.random.randint(10, 30), 'magnitude': np.random.uniform(2.0, 4.0)}

            self.configure_client_drift(cid, drift_type, **params)

    def generate_client_data(self, client_id: int,
                          n_samples: int = 500,
                          round_num: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """生成客户端数据"""
        if round_num is None:
            round_num = self.current_round

        dist = self.client_distributions[client_id]
        updated_dist = self._apply_drift(dist, round_num)
        X, y = self._sample_from_distribution(updated_dist, n_samples)

        return X, y

    def _apply_drift(self, dist: Dict, round_num: int) -> Dict:
        """应用数据漂移"""
        updated = {k: v.copy() if isinstance(v, np.ndarray) else v
                 for k, v in dist.items()}

        drift_type = dist['drift_type']
        params = dist['drift_params']

        if drift_type == 'gradual':
            drift_factor = round_num * 0.01
            for c in range(self.num_classes):
                drift = dist['drift_speed'] * drift_factor * 10
                updated['means'][c] = dist['means'][c] + drift

        elif drift_type == 'periodic':
            period = params.get('period', 10)
            amplitude = params.get('amplitude', 1.0)
            phase = 2 * np.pi * round_num / period
            for c in range(self.num_classes):
                periodic_drift = amplitude * np.sin(phase) * dist['drift_speed'] * 5
                updated['means'][c] = dist['means'][c] + periodic_drift

        elif drift_type == 'sudden':
            drift_round = params.get('drift_round', 20)
            magnitude = params.get('magnitude', 3.0)
            if round_num >= drift_round:
                sudden_change = magnitude * dist['drift_speed'] * 10
                for c in range(self.num_classes):
                    updated['means'][c] = dist['means'][c] + sudden_change

        if drift_type != 'none':
            class_weights = dist['class_weights'].copy()
            change = np.zeros_like(class_weights)
            change[:self.num_classes // 2] = round_num * 0.002
            change[self.num_classes // 2:] = -round_num * 0.002
            updated['class_weights'] = np.clip(class_weights + change, 0.01, 1.0)
            updated['class_weights'] /= updated['class_weights'].sum()

        return updated

    def _sample_from_distribution(self, dist: Dict,
                                n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """从分布中采样"""
        X_list = []
        y_list = []

        class_weights = dist['class_weights']
        class_counts = np.random.multinomial(n_samples, class_weights)

        for c in range(self.num_classes):
            n_c = class_counts[c]
            if n_c == 0:
                continue

            mean = dist['means'][c]
            cov = dist['covs'][c]

            X_c = np.random.multivariate_normal(mean, cov, n_c)
            y_c = np.full(n_c, c)

            X_list.append(X_c)
            y_list.append(y_c)

        X = np.vstack(X_list) if X_list else np.zeros((0, self.num_features))
        y = np.concatenate(y_list) if y_list else np.zeros(0, dtype=int)

        return X, y

    def simulate_round(self, round_num: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """模拟一轮所有客户端数据"""
        self.current_round = round_num

        all_data = {}
        for cid in range(self.num_clients):
            X, y = self.generate_client_data(cid, n_samples=300, round_num=round_num)
            all_data[cid] = (X, y)

        return all_data


def run_drift_demo():
    """运行漂移模拟演示"""
    print("=" * 60)
    print("Data Drift Simulator Demo")
    print("=" * 60)

    np.random.seed(42)

    simulator = DriftSimulator(num_clients=5, num_features=10, num_classes=5)

    drift_types = [
        (DriftType.GRADUAL, "Gradual Drift"),
        (DriftType.PERIODIC, "Periodic Drift"),
        (DriftType.SUDDEN, "Sudden Drift"),
        (DriftType.INCREMENTAL, "Incremental Drift")
    ]

    for drift_type, name in drift_types:
        print(f"\n{name}:")
        simulator.set_drift_type(drift_type)

        for round_num in [0, 10, 20]:
            X, y = simulator.generate_client_data(0, n_samples=200, round_num=round_num)
            stats = simulator.get_distribution_stats(X, y)
            print(f"  Round {round_num}: entropy={stats.get('class_entropy', 0):.3f}, "
                  f"feature_mean={stats.get('feature_mean', 0):.3f}")

    print("\nDrift Demo Complete!")


if __name__ == "__main__":
    run_drift_demo()