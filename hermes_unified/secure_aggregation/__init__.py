"""
Secure Aggregation Module

Implements Shamir Secret Sharing and Paillier Homomorphic Encryption
for privacy-preserving federated learning.
"""

from hermes_unified.secure_aggregation.shamir import (
    ShamirSecretSharing,
    SimplifiedShamirAggregator
)
from hermes_unified.secure_aggregation.paillier import (
    PaillierEncryption,
    PaillierArrayEncryption,
    create_demo_paillier
)
from hermes_unified.secure_aggregation.secure_aggregator import (
    SecureAggregator,
    SecureFLClient,
    run_secure_aggregation_demo
)

__all__ = [
    # Shamir
    'ShamirSecretSharing',
    'SimplifiedShamirAggregator',
    # Paillier
    'PaillierEncryption',
    'PaillierArrayEncryption',
    'create_demo_paillier',
    # Aggregator
    'SecureAggregator',
    'SecureFLClient',
    'run_secure_aggregation_demo'
]
