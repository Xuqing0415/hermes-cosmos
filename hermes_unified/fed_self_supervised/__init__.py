"""
Federated Self-Supervised Learning Module

Implements federated contrastive learning with:
- Global negative buffer for shared negative samples
- Distribution-aware temperature scaling
- Federated momentum encoder
- Privacy-preserving feature sharing
"""

from .contrastive_client import (
    ContrastiveClient,
    SimCLRClient,
    MoCoClient
)

from .global_buffer import (
    GlobalNegativeBuffer,
    PrivacyPreservingBuffer
)

from .momentum_encoder import (
    MomentumEncoder,
    FederatedMomentumEncoder
)

from .contrastive_loss import (
    NTXentLoss,
    MoCoLoss,
    BYOLLoss
)

from .fed_simclr import (
    FedSimCLRCoordinator,
    run_fed_simclr_demo
)

__all__ = [
    'ContrastiveClient',
    'SimCLRClient',
    'MoCoClient',
    'GlobalNegativeBuffer',
    'PrivacyPreservingBuffer',
    'MomentumEncoder',
    'FederatedMomentumEncoder',
    'NTXentLoss',
    'MoCoLoss',
    'BYOLLoss',
    'FedSimCLRCoordinator',
    'run_fed_simclr_demo'
]