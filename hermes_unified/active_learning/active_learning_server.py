"""
主动学习服务器
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AnnotationModel:
    """
    标注模型
    """

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def predict_labels(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测标签（伪标签）

        Returns:
            (predicted_labels, confidence_scores)
        """
        prototypes = np.random.randn(self.num_classes, embeddings.shape[1])

        distances = np.linalg.norm(embeddings[:, np.newaxis] - prototypes[np.newaxis], axis=2)

        labels = np.argmin(distances, axis=1)

        min_distances = np.min(distances, axis=1)
        confidence = 1 / (1 + min_distances)

        return labels, confidence


class ActiveLearningServer:
    """
    联邦主动学习服务器

    功能：
    1. 接收客户端选择的样本嵌入
    2. 生成/分发标注
    3. 聚合模型更新
    """

    def __init__(self, num_features: int,
                 num_classes: int = 10):
        self.num_features = num_features
        self.num_classes = num_classes

        self.global_weights = np.random.randn(num_features, num_classes) * 0.01
        self.global_bias = np.zeros(num_classes)

        self.annotation_model = AnnotationModel(num_classes)

        self.labeling_history = []

        logger.info("Active Learning Server initialized")

    def process_client_selection(self,
                                 client_selections: List[Dict]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """处理客户端的样本选择"""
        all_embeddings = []
        client_info = []

        for selection in client_selections:
            client_id = selection['client_id']
            embeddings = selection['embeddings']

            if embeddings is not None and len(embeddings) > 0:
                all_embeddings.append(embeddings)
                client_info.append({
                    'client_id': client_id,
                    'num_samples': len(embeddings),
                    'start_idx': len(all_embeddings) - 1
                })

        if not all_embeddings:
            return {}

        all_embeddings = np.vstack(all_embeddings)

        labels, confidences = self.annotation_model.predict_labels(all_embeddings)

        client_annotations = {}
        current_idx = 0

        for info in client_info:
            n = info['num_samples']
            client_annotations[info['client_id']] = (
                labels[current_idx:current_idx + n],
                confidences[current_idx:current_idx + n]
            )
            current_idx += n

        self.labeling_history.append({
            'num_samples_labeled': len(labels),
            'avg_confidence': np.mean(confidences)
        })

        return client_annotations

    def aggregate_models(self,
                       client_updates: List[Dict],
                       client_weights: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """聚合客户端模型更新"""
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
        """获取全局模型"""
        return {
            'weights': self.global_weights,
            'bias': self.global_bias
        }

    def get_labeling_stats(self) -> Dict[str, Any]:
        """获取标注统计"""
        if not self.labeling_history:
            return {'total_labeled': 0}

        return {
            'total_labeled': sum(h['num_samples_labeled'] for h in self.labeling_history),
            'avg_confidence': np.mean([h['avg_confidence'] for h in self.labeling_history])
        }


def run_active_learning_demo():
    """运行主动学习演示"""
    print("=" * 70)
    print("FEDERATED ACTIVE LEARNING DEMO")
    print("=" * 70)

    np.random.seed(42)

    num_clients = 3
    num_classes = 5
    num_features = 20
    num_rounds = 5

    server = ActiveLearningServer(num_features, num_classes)

    clients = []
    for cid in range(num_clients):
        from .active_learning_client import ActiveLearningClient
        client = ActiveLearningClient(cid, num_features, num_classes)

        n_labeled = 50
        X_labeled = np.random.randn(n_labeled, num_features)
        y_labeled = np.random.randint(0, num_classes, n_labeled)
        client.set_labeled_data(X_labeled, y_labeled)

        n_unlabeled = 500
        X_unlabeled = np.random.randn(n_unlabeled, num_features)
        client.add_unlabeled_pool(X_unlabeled)

        clients.append(client)

    print("\nFederated Active Learning Rounds:")

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        global_model = server.get_global_model()
        for client in clients:
            client.set_model_parameters(global_model['weights'], global_model['bias'])

        for client in clients:
            train_result = client.local_train()
            print(f"  Client {client.client_id}: loss={train_result['loss']:.4f}, acc={train_result['accuracy']:.4f}")

        budget_per_client = 20
        client_selections = []

        for client in clients:
            selection = client.select_samples_for_labeling(
                budget=budget_per_client,
                diversity_enhanced=True
            )
            client_selections.append(selection)
            print(f"  Client {client.client_id}: Selected {selection['num_selected']} samples")

        annotations = server.process_client_selection(client_selections)

        client_updates = [client.get_parameters() for client in clients]
        server.aggregate_models(client_updates)

        stats = server.get_labeling_stats()
        print(f"  Total labeled samples: {stats['total_labeled']}")

    print("\n" + "=" * 70)
    print("ACTIVE LEARNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_active_learning_demo()