"""
Self-Evolving Federated Learning Module

Enables Hermes to automatically run experiments, analyze results,
and self-improve in an infinite loop.

[!] IS_SIMULATION = True —— 本模块是**模拟**，不是真实联邦训练。
    `SelfEvolvingFederatedLearning.run_single_experiment` 用合成任务与带噪声的预测值
    代替真实 FL 训练，所以 evolution_progress.json / best_deployment.json 里的精度、
    轮次都只是演示数据。调用方、论文生成器、数据采集层在把它当成「系统能力」之前，
    必须先检查这个常量；详见同目录 MOCKED.md。
"""

from .self_evolution import SelfEvolvingFederatedLearning

IS_SIMULATION = True

__all__ = ["IS_SIMULATION", "SelfEvolvingFederatedLearning"]
