"""
Federated XAI: Client and Server for Federated Explainability

Implements client-side explanation generation and server-side aggregation
for federated learning models.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedXAIClient:
    """
    Federated learning client with XAI capabilities.
    
    Each client can:
    1. Generate local explanations (SHAP values) for its model
    2. Compute feature importance distributions
    3. Share aggregated importance statistics (not raw data)
    """
    
    def __init__(self, client_id: str, model, feature_names: Optional[List[str]] = None):
        """
        Initialize federated XAI client.
        
        Args:
            client_id: Unique client identifier
            model: Trained local model with predict method
            feature_names: Names of features
        """
        self.client_id = client_id
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(model.n_features if hasattr(model, 'n_features') else 10)]
        
        # Local explanation cache
        self.local_explanations = []
        self.feature_importance_history = []
        
        # Anomaly detection baseline
        self.baseline_importance = None
        self.importance_history = []
    
    def generate_local_explanation(self, X: np.ndarray, method: str = 'shap',
                                  n_samples: int = 100) -> Dict[str, Any]:
        """
        Generate local explanation for the model.
        
        Args:
            X: Local test data
            method: Explanation method ('shap', 'lime', 'permutation')
            n_samples: Number of samples for explanation
        
        Returns:
            Explanation dictionary with feature importance
        """
        from .shap_explainer import KernelSHAPExplainer, compute_feature_importance
        
        logger.info(f"Client {self.client_id}: Generating {method} explanation")
        
        if method == 'shap':
            explainer = KernelSHAPExplainer(background_data=X, nsamples=n_samples)
            
            # Sample a few instances to explain
            sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)
            X_sample = X[sample_indices]
            
            results = []
            for x in X_sample:
                result = explainer.explain(self.model.predict, x, self.feature_names)
                results.append(result)
            
            # Aggregate local explanations
            shap_values = np.array([r['shap_values'] for r in results])
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            explanation = {
                'method': 'shap',
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'feature_names': self.feature_names,
                'n_explained': len(sample_indices)
            }
            
        elif method == 'lime':
            from .shap_explainer import LimeExplainer
            
            explainer = LimeExplainer(num_features=len(self.feature_names))
            
            x_sample = X[0]
            result = explainer.explain_instance(self.model.predict, x_sample, self.feature_names)
            
            explanation = {
                'method': 'lime',
                'feature_weights': result['feature_weights'],
                'top_features': result['top_features'],
                'feature_names': self.feature_names
            }
            
        else:  # permutation
            y = np.random.randint(0, 2, len(X))  # Dummy labels
            result = compute_feature_importance(self.model, X, y, method='permutation', 
                                              feature_names=self.feature_names)
            explanation = {
                'method': 'permutation',
                'feature_importance': result['importance'],
                'feature_names': self.feature_names
            }
        
        self.local_explanations.append(explanation)
        
        if 'feature_importance' in explanation:
            self.feature_importance_history.append(explanation['feature_importance'])
        
        return explanation
    
    def get_feature_importance_vector(self) -> np.ndarray:
        """
        Get aggregated feature importance vector for sharing.
        
        Returns:
            Mean absolute feature importance across local explanations
        """
        if not self.local_explanations:
            return np.zeros(len(self.feature_names))
        
        importances = []
        
        for exp in self.local_explanations:
            if 'feature_importance' in exp:
                importances.append(exp['feature_importance'])
        
        if not importances:
            return np.zeros(len(self.feature_names))
        
        return np.mean(importances, axis=0)
    
    def set_baseline(self, importance_vector: np.ndarray):
        """Set baseline importance for anomaly detection."""
        self.baseline_importance = importance_vector
    
    def compute_importance_deviation(self) -> float:
        """
        Compute deviation from baseline importance.
        
        Used for detecting:
        - Data distribution drift
        - Potential malicious behavior
        - Model degradation
        
        Returns:
            Maximum absolute deviation from baseline
        """
        if self.baseline_importance is None:
            return 0.0
        
        current = self.get_feature_importance_vector()
        
        if len(current) != len(self.baseline_importance):
            return 0.0
        
        deviation = np.abs(current - self.baseline_importance)
        max_deviation = np.max(deviation)
        
        self.importance_history.append({
            'deviation': max_deviation,
            'timestamp': time.time()
        })
        
        return float(max_deviation)
    
    def is_anomalous(self, threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Check if client is exhibiting anomalous behavior.
        
        Args:
            threshold: Deviation threshold for anomaly detection
        
        Returns:
            (is_anomalous, deviation_score)
        """
        deviation = self.compute_importance_deviation()
        return deviation > threshold, deviation
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client XAI statistics."""
        return {
            'client_id': self.client_id,
            'n_explanations': len(self.local_explanations),
            'feature_names': self.feature_names,
            'latest_importance': self.get_feature_importance_vector().tolist(),
            'current_deviation': self.compute_importance_deviation() if self.baseline_importance is not None else 0.0
        }


class FederatedXAIServer:
    """
    Federated XAI server that aggregates explanations from clients.
    
    Collects feature importance vectors from clients and computes:
    - Global average importance
    - Cross-client variance
    - Per-class importance
    - Anomaly detection alerts
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize federated XAI server.
        
        Args:
            feature_names: Names of features
        """
        self.feature_names = feature_names or []
        self.clients: Dict[str, FederatedXAIClient] = {}
        
        # Aggregated statistics
        self.global_importance = None
        self.importance_history = []
        self.anomaly_alerts = []
        
        # Per-round history
        self.round_history = []
    
    def register_client(self, client: FederatedXAIClient):
        """Register a client with the XAI server."""
        self.clients[client.client_id] = client
    
    def collect_importance_vectors(self) -> Dict[str, np.ndarray]:
        """Collect feature importance vectors from all clients."""
        vectors = {}
        
        for client_id, client in self.clients.items():
            vectors[client_id] = client.get_feature_importance_vector()
        
        return vectors
    
    def aggregate_importance(self, vectors: Dict[str, np.ndarray]
                           ) -> Dict[str, np.ndarray]:
        """
        Aggregate feature importance vectors from clients.
        
        Args:
            vectors: Dictionary mapping client_id to importance vector
        
        Returns:
            Aggregated statistics
        """
        if not vectors:
            return {}
        
        client_ids = list(vectors.keys())
        importance_matrix = np.array([vectors[cid] for cid in client_ids])
        
        mean_importance = np.mean(importance_matrix, axis=0)
        std_importance = np.std(importance_matrix, axis=0)
        var_importance = np.var(importance_matrix, axis=0)
        
        # Update global importance
        self.global_importance = mean_importance
        
        # Store in history
        self.importance_history.append({
            'round': len(self.round_history),
            'mean': mean_importance,
            'std': std_importance,
            'variance': var_importance
        })
        
        # Set baseline for new clients
        for client_id, client in self.clients.items():
            if client.baseline_importance is None:
                client.set_baseline(mean_importance)
        
        return {
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'variance_importance': var_importance,
            'client_importance': vectors,
            'n_clients': len(vectors)
        }
    
    def detect_anomalies(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect anomalous clients based on feature importance deviation.
        
        Args:
            threshold: Deviation threshold
        
        Returns:
            List of anomaly alerts
        """
        alerts = []
        
        for client_id, client in self.clients.items():
            is_anomalous, deviation = client.is_anomalous(threshold)
            
            if is_anomalous:
                alert = {
                    'client_id': client_id,
                    'deviation': deviation,
                    'threshold': threshold,
                    'timestamp': time.time(),
                    'severity': 'high' if deviation > 0.5 else 'medium'
                }
                alerts.append(alert)
                self.anomaly_alerts.append(alert)
        
        return alerts
    
    def get_per_class_importance(self, X: np.ndarray, y: np.ndarray,
                                class_labels: List[int]) -> Dict[int, np.ndarray]:
        """
        Compute per-class feature importance.
        
        Args:
            X: Feature data
            y: Labels
            class_labels: List of class labels to compute importance for
        
        Returns:
            Dictionary mapping class to importance vector
        """
        per_class_importance = {}
        
        for class_label in class_labels:
            class_mask = y == class_label
            X_class = X[class_mask]
            
            if len(X_class) == 0:
                continue
            
            # Generate explanation for this class
            from .shap_explainer import KernelSHAPExplainer
            
            # Get a sample model prediction
            predictions = np.array([self.clients[c].model.predict(X_class[:10]) 
                                   for c in self.clients])
            avg_pred = np.mean(predictions, axis=0)
            
            # Compute simple importance (using prediction gradient approximation)
            importance = np.zeros(X.shape[1])
            
            for i in range(min(50, len(X_class))):
                x = X_class[i]
                pred = self.clients[list(self.clients.keys())[0]].model.predict(x.reshape(1, -1))
                
                for j in range(X.shape[1]):
                    x_plus = x.copy()
                    x_plus[j] += 0.01
                    pred_plus = self.clients[list(self.clients.keys())[0]].model.predict(x_plus.reshape(1, -1))
                    importance[j] += abs(pred_plus - pred)
            
            importance /= len(X_class)
            per_class_importance[class_label] = importance
        
        return per_class_importance
    
    def run_xai_round(self, local_explanations: bool = True,
                     detect_anomalies: bool = True,
                     anomaly_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Run one XAI aggregation round.
        
        Args:
            local_explanations: Whether clients generate local explanations
            detect_anomalies: Whether to run anomaly detection
            anomaly_threshold: Threshold for anomaly detection
        
        Returns:
            Round results
        """
        # Collect importance vectors
        vectors = self.collect_importance_vectors()
        
        # Aggregate
        aggregation = self.aggregate_importance(vectors)
        
        # Detect anomalies
        anomaly_results = []
        if detect_anomalies:
            anomaly_results = self.detect_anomalies(anomaly_threshold)
        
        result = {
            'round': len(self.round_history) + 1,
            'aggregation': aggregation,
            'anomalies': anomaly_results,
            'n_clients': len(self.clients)
        }
        
        self.round_history.append(result)
        
        return result
    
    def get_global_explanation(self) -> Dict[str, Any]:
        """
        Get global model explanation.
        
        Returns:
            Global feature importance and statistics
        """
        if self.global_importance is None:
            return {'error': 'No explanations collected yet'}
        
        # Sort features by importance
        if self.feature_names:
            importance_pairs = list(zip(self.feature_names, self.global_importance))
        else:
            importance_pairs = [(f"feature_{i}", v) 
                              for i, v in enumerate(self.global_importance)]
        
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'global_importance': self.global_importance.tolist(),
            'top_features': importance_pairs[:10],
            'total_features': len(self.global_importance),
            'n_rounds': len(self.round_history),
            'n_anomalies': len(self.anomaly_alerts)
        }
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for the visualization dashboard.
        
        Returns:
            Dashboard data including all visualizations
        """
        # Client importance matrix for heatmap
        client_importance = {}
        for client_id, vector in self.collect_importance_vectors().items():
            client_importance[client_id] = vector.tolist()
        
        # Importance history for line chart
        history_data = []
        for h in self.importance_history:
            history_data.append({
                'round': h['round'],
                'mean': h['mean'].tolist(),
                'std': h['std'].tolist()
            })
        
        # Anomaly alerts
        alerts_data = [{
            'client_id': a['client_id'],
            'deviation': a['deviation'],
            'severity': a['severity'],
            'timestamp': a['timestamp']
        } for a in self.anomaly_alerts[-10:]]  # Last 10 alerts
        
        return {
            'feature_names': self.feature_names,
            'client_importance': client_importance,
            'history': history_data,
            'alerts': alerts_data,
            'global_importance': self.global_importance.tolist() if self.global_importance is not None else None,
            'total_clients': len(self.clients),
            'total_anomalies': len(self.anomaly_alerts)
        }


if __name__ == "__main__":
    print("=== Testing Federated XAI ===\n")
    
    # Create dummy clients
    class DummyModel:
        def __init__(self, n_features):
            self.n_features = n_features
            self.weights = np.random.randn(n_features)
        
        def predict(self, X):
            return X @ self.weights
    
    server = FederatedXAIServer(feature_names=[f"feature_{i}" for i in range(10)])
    
    for i in range(5):
        model = DummyModel(n_features=10)
        client = FederatedXAIClient(f"client_{i}", model, server.feature_names)
        server.register_client(client)
    
    # Generate some explanations
    for _ in range(3):
        X = np.random.randn(50, 10)
        
        for client in server.clients.values():
            exp = client.generate_local_explanation(X, method='shap')
    
    # Run XAI round
    result = server.run_xai_round()
    
    print(f"Round {result['round']}:")
    print(f"  Clients: {result['n_clients']}")
    print(f"  Top features: {result['aggregation']['mean_importance'][:3]}")
    print(f"  Anomalies: {len(result['anomalies'])}")
    
    # Get global explanation
    global_exp = server.get_global_explanation()
    print(f"\nGlobal explanation top features:")
    for name, imp in global_exp['top_features'][:5]:
        print(f"  {name}: {imp:.4f}")
