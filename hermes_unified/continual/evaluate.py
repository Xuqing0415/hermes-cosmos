"""
连续学习评估指标
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ContinualLearningEvaluator:
    """
    连续学习评估器
    """

    def __init__(self):
        self.accuracy_history = {}

    def record_accuracy(self, round_num: int,
                      client_id: int,
                      accuracy: float,
                      is_baseline: bool = False):
        """记录准确率"""
        if client_id not in self.accuracy_history:
            self.accuracy_history[client_id] = []

        self.accuracy_history[client_id].append({
            'round': round_num,
            'accuracy': accuracy
        })

    def compute_forgetting_rate(self, client_id: int) -> float:
        """计算遗忘率"""
        if client_id not in self.accuracy_history:
            return 0.0

        history = self.accuracy_history[client_id]
        if len(history) < 2:
            return 0.0

        accuracies = [h['accuracy'] for h in history]
        best_acc = max(accuracies[:-1]) if len(accuracies) > 1 else accuracies[0]
        final_acc = accuracies[-1]

        return max(0, best_acc - final_acc)

    def compute_average_forgetting(self) -> float:
        """计算所有客户端的平均遗忘率"""
        forgetting_rates = [
            self.compute_forgetting_rate(cid)
            for cid in self.accuracy_history.keys()
        ]

        return np.mean(forgetting_rates) if forgetting_rates else 0.0

    def compute_fairness(self) -> Dict[str, float]:
        """计算个体公平性"""
        forgetting_rates = [
            self.compute_forgetting_rate(cid)
            for cid in self.accuracy_history.keys()
        ]

        if not forgetting_rates:
            return {'gini': 0.0, 'variance': 0.0}

        forgetting_array = np.array(forgetting_rates)

        sorted_f = np.sort(forgetting_array)
        n = len(sorted_f)
        cumsum = np.cumsum(sorted_f)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_f))) / (n * np.sum(sorted_f))) - (n + 1) / n

        variance = np.var(forgetting_array)

        return {
            'gini': gini,
            'variance': variance,
            'max_forgetting': np.max(forgetting_array),
            'min_forgetting': np.min(forgetting_array)
        }

    def compute_summary_metrics(self) -> Dict[str, float]:
        """计算汇总指标"""
        return {
            'avg_forgetting': self.compute_average_forgetting(),
            'max_forgetting': max([
                self.compute_forgetting_rate(cid)
                for cid in self.accuracy_history.keys()
            ]) if self.accuracy_history else 0.0,
            **self.compute_fairness()
        }

    def print_report(self):
        """打印评估报告"""
        summary = self.compute_summary_metrics()

        print("\n" + "=" * 60)
        print("CONTINUAL LEARNING EVALUATION REPORT")
        print("=" * 60)

        print(f"\nSummary Metrics:")
        print(f"  Average Forgetting: {summary['avg_forgetting']:.4f}")
        print(f"  Maximum Forgetting: {summary['max_forgetting']:.4f}")
        print(f"  Forgetting Gini: {summary['gini']:.4f}")
        print(f"  Forgetting Variance: {summary['variance']:.4f}")

        print(f"\nPer-Client Forgetting Rates:")
        for cid in sorted(self.accuracy_history.keys()):
            fr = self.compute_forgetting_rate(cid)
            print(f"  Client {cid}: {fr:.4f}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Continual Learning Evaluator")
    evaluator = ContinualLearningEvaluator()

    for round_num in range(10):
        for client_id in range(5):
            if round_num < 5:
                acc = 0.5 + round_num * 0.08 + np.random.randn() * 0.02
            else:
                acc = 0.9 - (round_num - 5) * 0.06 + np.random.randn() * 0.02

            evaluator.record_accuracy(round_num, client_id, acc)

    evaluator.print_report()