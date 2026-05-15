"""
Self-Evolving Federated Learning Module

Enables Hermes to automatically run experiments, analyze results,
and self-improve in an infinite loop.
"""

from .self_evolution import SelfEvolvingFederatedLearning

__all__ = ['SelfEvolvingFederatedLearning']
