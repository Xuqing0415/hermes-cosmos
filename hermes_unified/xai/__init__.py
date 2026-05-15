"""
FedXAI: Federated Explainable AI Module

Provides model interpretability for federated learning using SHAP values
and feature importance aggregation.
"""

from .shap_explainer import KernelSHAPExplainer, LimeExplainer, compute_feature_importance
from .federated_xai import FederatedXAIClient, FederatedXAIServer
from .dashboard import FedXAIDashboard, run_xai_dashboard, dashboard

__all__ = [
    'KernelSHAPExplainer',
    'LimeExplainer',
    'compute_feature_importance',
    'FederatedXAIClient',
    'FederatedXAIServer',
    'FedXAIDashboard',
    'run_xai_dashboard',
    'dashboard'
]
