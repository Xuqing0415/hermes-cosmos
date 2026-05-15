"""
SHAP Explainer for Federated Learning

Implements KernelSHAP for local explanation of federated learning models.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KernelSHAPExplainer:
    """
    KernelSHAP explainer for federated learning models.
    
    KernelSHAP is a model-agnostic explanation method based on
    Shapley values. It estimates the contribution of each feature
    to a prediction by sampling subsets of features.
    
    For federated learning, we use a simplified version that:
    1. Works with numpy models (no need for PyTorch/TF graphs)
    2. Can be computed locally on each client
    3. Outputs aggregated feature importance vectors for sharing
    """
    
    def __init__(self, background_data: Optional[np.ndarray] = None,
                 nsamples: int = 100, random_state: int = 42):
        """
        Initialize KernelSHAP explainer.
        
        Args:
            background_data: Background dataset for computing conditional expectations
            nsamples: Number of samples for SHAP approximation
            random_state: Random seed for reproducibility
        """
        self.background = background_data
        self.nsamples = nsamples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def explain(self, model, X: np.ndarray, feature_names: Optional[List[str]] = None
               ) -> Dict[str, np.ndarray]:
        """
        Generate SHAP values for a single prediction.
        
        Args:
            model: Prediction function (takes numpy array, returns predictions)
            X: Input sample to explain (1D array)
            feature_names: Names of features
        
        Returns:
            Dictionary with SHAP values and metadata
        """
        n_features = len(X)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Compute SHAP values using linear approximation
        shap_values = self._compute_shap_values(model, X)
        
        return {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'expected_value': float(np.mean(self.background @ np.zeros(n_features)) if self.background is not None else 0),
            'n_features': n_features
        }
    
    def explain_batch(self, model, X: np.ndarray, feature_names: Optional[List[str]] = None
                     ) -> Dict[str, np.ndarray]:
        """
        Generate SHAP values for multiple predictions.
        
        Args:
            model: Prediction function
            X: Input samples (2D array, n_samples x n_features)
            feature_names: Names of features
        
        Returns:
            Dictionary with SHAP values array and metadata
        """
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Compute SHAP for each sample
        shap_values = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            shap_values[i] = self._compute_shap_values(model, X[i])
        
        # Compute feature importance (mean absolute SHAP values)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'n_samples': n_samples,
            'n_features': n_features
        }
    
    def _compute_shap_values(self, model, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a single sample using linear approximation.
        
        This simplified version uses a linear model to approximate SHAP values:
        f(x) ≈ f(x_base) + Σ φ_i * (x_i - x_base_i)
        
        where φ_i are the SHAP values.
        """
        n_features = len(X)
        
        # Generate random subsets for estimation
        phi = np.zeros(n_features)
        
        for _ in range(self.nsamples):
            # Random subset mask
            subset = np.random.binomial(1, 0.5, n_features)
            
            # Compute marginal contribution
            X_subset = X * subset
            
            # Weight based on subset size (Shapley kernel)
            subset_size = np.sum(subset)
            if subset_size == 0 or subset_size == n_features:
                continue
            
            weight = (n_features - 1) / (subset_size * (n_features - subset_size))
            
            # Predict with subset
            pred_subset = self._predict_sample(model, X_subset)
            
            # Estimate SHAP value contribution
            phi += weight * pred_subset
        
        phi /= self.nsamples
        
        return phi
    
    def _predict_sample(self, model, X: np.ndarray) -> float:
        """Make prediction for a single sample."""
        X = X.reshape(1, -1)
        
        if callable(model):
            pred = model(X)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            return float(pred.flatten()[0] if hasattr(pred, 'flatten') else pred)
        
        return 0.0
    
    def compute_aggregated_importance(self, shap_results: List[Dict[str, Any]]
                                     ) -> Dict[str, np.ndarray]:
        """
        Aggregate SHAP values from multiple clients.
        
        Args:
            shap_results: List of SHAP result dictionaries
        
        Returns:
            Aggregated importance statistics
        """
        all_importances = []
        
        for result in shap_results:
            if 'feature_importance' in result:
                all_importances.append(result['feature_importance'])
            elif 'shap_values' in result:
                all_importances.append(np.mean(np.abs(result['shap_values']), axis=0))
        
        if not all_importances:
            return {}
        
        all_importances = np.array(all_importances)
        
        return {
            'mean_importance': np.mean(all_importances, axis=0),
            'std_importance': np.std(all_importances, axis=0),
            'min_importance': np.min(all_importances, axis=0),
            'max_importance': np.max(all_importances, axis=0),
            'client_variance': np.var(all_importances, axis=0)  # Cross-client variance
        }


def compute_feature_importance(model, X: np.ndarray, y: np.ndarray,
                               method: str = 'permutation',
                               feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Compute feature importance for federated learning models.
    
    Supports multiple methods:
    - 'permutation': Permutation importance
    - 'gradient': Gradient-based importance
    - 'random': Random baseline
    
    Args:
        model: Trained model with predict method
        X: Input features
        y: Target labels
        method: Importance computation method
        feature_names: Names of features
    
    Returns:
        Feature importance scores
    """
    n_features = X.shape[1]
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    if method == 'permutation':
        baseline_pred = model.predict(X)
        baseline_score = np.mean((baseline_pred.flatten() - y) ** 2)
        
        importance = np.zeros(n_features)
        
        for i in range(n_features):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            permuted_pred = model.predict(X_permuted)
            permuted_score = np.mean((permuted_pred.flatten() - y) ** 2)
            
            importance[i] = permuted_score - baseline_score
        
    elif method == 'gradient':
        importance = np.zeros(n_features)
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            
            for i in range(n_features):
                grad = np.gradient(proba[:, 1], X[:, i])
                importance[i] = np.mean(np.abs(grad))
        else:
            predictions = model.predict(X)
            
            for i in range(n_features):
                grad = np.gradient(predictions, X[:, i])
                importance[i] = np.mean(np.abs(grad))
    
    else:  # random baseline
        importance = np.random.rand(n_features)
        importance /= np.sum(importance)
    
    # Normalize
    importance = np.abs(importance)
    if np.sum(importance) > 0:
        importance /= np.sum(importance)
    
    return {
        'importance': importance,
        'feature_names': feature_names,
        'method': method
    }


class LimeExplainer:
    """
    Simple LIME (Local Interpretable Model-agnostic Explanations) implementation.
    
    LIME explains predictions by approximating the model locally with
    an interpretable model (e.g., linear regression) on perturbed samples.
    """
    
    def __init__(self, num_features: int = 10, num_samples: int = 1000):
        """
        Initialize LIME explainer.
        
        Args:
            num_features: Number of features to include in explanation
            num_samples: Number of perturbed samples to generate
        """
        self.num_features = num_features
        self.num_samples = num_samples
    
    def explain_instance(self, model, X: np.ndarray, feature_names: Optional[List[str]] = None
                       ) -> Dict[str, Any]:
        """
        Explain a single instance.
        
        Args:
            model: Prediction function
            X: Input instance (1D array)
            feature_names: Feature names
        
        Returns:
            Explanation with feature weights
        """
        n_features = len(X)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Generate perturbed samples
        perturbed_samples = []
        perturbations = []
        
        for _ in range(self.num_samples):
            # Binary mask for presence/absence of features
            mask = np.random.binomial(1, 0.5, n_features)
            perturbations.append(mask)
            
            # Perturbed instance
            X_perturbed = X * mask
            perturbed_samples.append(X_perturbed)
        
        perturbed_samples = np.array(perturbed_samples)
        
        # Get predictions for perturbed samples
        predictions = []
        for X_p in perturbed_samples:
            pred = model.predict(X_p.reshape(1, -1))
            if hasattr(pred, 'flatten'):
                pred = pred.flatten()[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Simple weighted linear regression to get feature importance
        perturbations = np.array(perturbations)
        
        # Weight by proximity to original instance
        distances = np.sum(np.abs(perturbations - 1), axis=1)
        weights = np.exp(-distances ** 2 / (n_features ** 2))
        
        # Solve weighted least squares
        X_design = perturbations
        y = predictions - predictions[0]  # Center around original prediction
        
        # Simple approximation: weighted mean
        importance = np.zeros(n_features)
        for i in range(n_features):
            mask_i = perturbations[:, i] == 1
            if np.sum(mask_i) > 0:
                importance[i] = np.average(y[mask_i], weights=weights[mask_i])
        
        # Get top features
        top_indices = np.argsort(np.abs(importance))[-self.num_features:][::-1]
        
        return {
            'feature_weights': importance,
            'top_features': [(feature_names[i], importance[i]) for i in top_indices],
            'feature_names': feature_names,
            'local_prediction': float(predictions[0])
        }


if __name__ == "__main__":
    print("=== Testing SHAP Explainer ===\n")
    
    # Simple test with a linear model
    np.random.seed(42)
    
    def linear_model(X):
        return X @ np.array([0.5, 0.3, -0.2, 0.1, 0.4]) + 0.1
    
    X_train = np.random.randn(100, 5)
    X_test = np.array([[1.0, 0.5, -0.5, 0.2, 0.1]])
    
    explainer = KernelSHAPExplainer(background_data=X_train, nsamples=100)
    
    result = explainer.explain(linear_model, X_test[0], feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])
    
    print("SHAP Values:")
    for name, value in zip(result['feature_names'], result['shap_values']):
        print(f"  {name}: {value:.4f}")
