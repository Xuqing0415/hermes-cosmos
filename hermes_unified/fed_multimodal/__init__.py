"""
Federated Multimodal Learning Module

Implements federated learning for multi-modal data, supporting:
- Modality-aware federated architecture
- Cross-modal alignment and fusion
- Missing modality handling
- Differential privacy for different modalities
"""

from .multimodal_client import (
    MultimodalFedClient,
    ModalityEncoder,
    SharedPrivateEncoder
)

from .multimodal_server import (
    MultimodalFedServer,
    ModalityAggregator
)

from .modality_aligner import (
    CrossModalAligner,
    ContrastiveAligner,
    CLIPStyleAligner
)

from .missing_modality_handler import (
    MissingModalityHandler,
    ModalityMasker,
    KnowledgeDistillationFiller,
    VirtualModalityGenerator
)

from .data_utils import (
    generate_synthetic_multimodal_data,
    MultimodalDataset,
    split_modalities_across_clients,
    create_dataloader
)

from .multimodal_coordinator import (
    MultimodalFedCoordinator,
    run_multimodal_demo
)

from .multimodal_client import (
    ImageEncoder,
    TextEncoder,
    TabularEncoder,
    MultimodalFusion
)

from .multimodal_server import (
    DifferentialPrivacyBudget,
    SensitivityClipper
)

from .modality_aligner import (
    FederatedAlignerServer
)

__all__ = [
    'MultimodalFedClient',
    'ModalityEncoder',
    'SharedPrivateEncoder',
    'ImageEncoder',
    'TextEncoder',
    'TabularEncoder',
    'MultimodalFusion',
    'MultimodalFedServer',
    'ModalityAggregator',
    'DifferentialPrivacyBudget',
    'SensitivityClipper',
    'CrossModalAligner',
    'ContrastiveAligner',
    'CLIPStyleAligner',
    'FederatedAlignerServer',
    'MissingModalityHandler',
    'ModalityMasker',
    'KnowledgeDistillationFiller',
    'VirtualModalityGenerator',
    'generate_synthetic_multimodal_data',
    'MultimodalDataset',
    'split_modalities_across_clients',
    'create_dataloader',
    'MultimodalFedCoordinator',
    'run_multimodal_demo'
]