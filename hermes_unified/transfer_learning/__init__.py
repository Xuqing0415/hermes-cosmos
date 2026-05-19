"""
Federated Transfer Learning Module

Submodules:
- domain_adaptation: Domain adaptation techniques
- knowledge_distillation: Knowledge distillation methods
- zero_shot: Zero-shot transfer learning
"""

from .domain_adaptation import (
    MMDLoss,
    DomainDiscriminator,
    DomainAdversarialAdaptor,
    DomainAdaptationServer,
    DomainAdaptationClient
)

from .knowledge_distillation import (
    FedDistillationServer,
    FedDistillationClient,
    DistillationLoss,
    TeacherStudentTrainer
)

from .zero_shot import (
    PromptAdaptor,
    ZeroShotTransfer,
    DomainPromptGenerator
)

__all__ = [
    'MMDLoss',
    'DomainDiscriminator',
    'DomainAdversarialAdaptor',
    'DomainAdaptationServer',
    'DomainAdaptationClient',
    'FedDistillationServer',
    'FedDistillationClient',
    'DistillationLoss',
    'TeacherStudentTrainer',
    'PromptAdaptor',
    'ZeroShotTransfer',
    'DomainPromptGenerator'
]