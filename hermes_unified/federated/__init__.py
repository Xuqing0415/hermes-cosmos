"""
Hermes Federated Learning Module

Provides components for federated learning integration:
- Client selection strategies
- Secure aggregation protocols
- Federated learning simulator
- Attack and defense mechanisms
"""

from .client_selector import ClientSelector, ClientSampler
from .secure_aggregator import SecureAggregator, FedAvgAggregator, DecentralizedAggregator
from .federated_simulator import (
    FederatedSimulator, FLClient, NonIIDDataGenerator,
    LabelFlipAttackClient, GradientScaleAttackClient, BackdoorAttackClient,
    KrumServer, TrimmedMeanServer
)
from .federated_server import FederatedServer
from .federated_client import FederatedClient
from .attacks import AttackClient, AttackType, AttackManager
from .defenses import DefenseServer, DefenseType
from .attack_defense_simulator import AttackDefenseSimulator
from .attack_evolution import AttackEvolution
from .adaptive_defense import AdaptiveDefenseServer, OnlineAttackEvolution, AttackDefenseGame

__all__ = [
    'ClientSelector',
    'ClientSampler',
    'SecureAggregator',
    'FedAvgAggregator',
    'DecentralizedAggregator',
    'FederatedSimulator',
    'FLClient',
    'NonIIDDataGenerator',
    'LabelFlipAttackClient',
    'GradientScaleAttackClient',
    'BackdoorAttackClient',
    'KrumServer',
    'TrimmedMeanServer',
    'FederatedServer',
    'FederatedClient',
    'AttackClient',
    'AttackType',
    'AttackManager',
    'DefenseServer',
    'DefenseType',
    'AttackDefenseSimulator',
    'AttackEvolution',
    'AdaptiveDefenseServer',
    'OnlineAttackEvolution',
    'AttackDefenseGame'
]
