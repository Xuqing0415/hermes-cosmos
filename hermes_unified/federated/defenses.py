"""
Federated Learning Defense Module

Implements various defense strategies against attacks.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class DefenseType:
    """Enumeration of defense types"""
    KRUM = 'krum'
    TRIMMED_MEAN = 'trimmed_mean'
    ANOMALY_DETECTION = 'anomaly_detection'
    FLTRUST = 'fltrust'
    DIFFERENTIAL_PRIVACY = 'differential_privacy'


class DefenseServer:
    """
    Defense server that implements various defense strategies.
    
    Supported defenses:
    - Krum: Select update closest to others
    - Trimmed Mean: Remove outliers before averaging
    - Anomaly Detection: Detect and filter abnormal updates
    - FLTrust: Compare updates with trusted baseline
    - Differential Privacy: Add noise to prevent gradient inversion
    """
    
    def __init__(self, defense_type: str = 'trimmed_mean', **kwargs):
        """
        Initialize defense server.
        
        Args:
            defense_type: Type of defense to use
            kwargs: Additional defense parameters
        """
        self.defense_type = defense_type.lower()
        self.kwargs = kwargs
        
        # Defense parameters
        self.trim_ratio = kwargs.get('trim_ratio', 0.2)  # For trimmed mean
        self.krum_k = kwargs.get('krum_k', 1)  # For Krum
        self.anomaly_threshold = kwargs.get('anomaly_threshold', 3.0)  # For anomaly detection
        self.noise_scale = kwargs.get('noise_scale', 0.01)  # For differential privacy
        
        # Trusted data for FLTrust
        self.trusted_data = None
        self.trusted_model = None
        
        # Statistics
        self.defense_log = []
        self.malicious_updates_filtered = 0
    
    def set_trusted_data(self, X: np.ndarray, y: np.ndarray):
        """Set trusted data for FLTrust."""
        self.trusted_data = (X, y)
    
    def _krum(self, updates: List[np.ndarray], k: int = 1) -> np.ndarray:
        """
        Krum defense: Select update with smallest distance to others.
        
        Args:
            updates: List of client updates
            k: Number of updates to select
            
        Returns:
            Selected update
        """
        if len(updates) <= k:
            return np.mean(updates, axis=0)
        
        # Compute distances between all pairs
        n = len(updates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = np.linalg.norm(updates[i] - updates[j])
        
        # For each update, sum distances to k closest others
        scores = []
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[:k])
            scores.append(score)
        
        # Select update with smallest score
        selected_idx = np.argmin(scores)
        return updates[selected_idx]
    
    def _trimmed_mean(self, updates: List[np.ndarray], trim_ratio: float = 0.2) -> np.ndarray:
        """
        Trimmed mean defense: Remove outliers before averaging.
        
        Args:
            updates: List of client updates
            trim_ratio: Fraction of updates to trim from each end
            
        Returns:
            Trimmed mean update
        """
        if len(updates) <= 2:
            return np.mean(updates, axis=0)
        
        n = len(updates)
        n_trim = int(n * trim_ratio)
        
        if n_trim == 0:
            return np.mean(updates, axis=0)
        
        # Compute norms and sort
        norms = np.array([np.linalg.norm(u) for u in updates])
        sorted_indices = np.argsort(norms)
        
        # Trim from both ends
        keep_indices = sorted_indices[n_trim:-n_trim]
        
        if len(keep_indices) == 0:
            keep_indices = sorted_indices[:1]
        
        # Compute mean of remaining updates
        kept_updates = [updates[i] for i in keep_indices]
        return np.mean(kept_updates, axis=0)
    
    def _anomaly_detection(self, updates: List[np.ndarray], threshold: float = 3.0) -> Tuple[List[np.ndarray], int]:
        """
        Anomaly detection: Filter updates based on norm and cosine similarity.
        
        Args:
            updates: List of client updates
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Tuple of (filtered_updates, num_filtered)
        """
        if len(updates) <= 1:
            return updates, 0
        
        # Compute norms
        norms = np.array([np.linalg.norm(u) for u in updates])
        
        # Z-score for norms
        norm_mean = np.mean(norms)
        norm_std = np.std(norms) if len(norms) > 1 else 1.0
        
        # Compute cosine similarity to mean
        mean_update = np.mean(updates, axis=0)
        mean_norm = np.linalg.norm(mean_update)
        
        similarities = []
        for u in updates:
            if mean_norm > 0 and np.linalg.norm(u) > 0:
                sim = np.dot(u.flatten(), mean_update.flatten()) / (np.linalg.norm(u) * mean_norm)
            else:
                sim = 0.0
            similarities.append(sim)
        
        sim_mean = np.mean(similarities)
        sim_std = np.std(similarities) if len(similarities) > 1 else 1.0
        
        # Filter anomalies
        filtered = []
        filtered_count = 0
        
        for i, u in enumerate(updates):
            z_norm = abs(norms[i] - norm_mean) / (norm_std + 1e-10)
            z_sim = abs(similarities[i] - sim_mean) / (sim_std + 1e-10)
            
            if z_norm < threshold and z_sim < threshold:
                filtered.append(u)
            else:
                filtered_count += 1
        
        if not filtered:
            filtered = [updates[np.argmin(norms)]]
        
        return filtered, filtered_count
    
    def _fltrust(self, updates: List[np.ndarray], global_model: np.ndarray) -> np.ndarray:
        """
        FLTrust defense: Compare updates with trusted baseline.
        
        Args:
            updates: List of client updates
            global_model: Current global model
            
        Returns:
            Filtered aggregated update
        """
        if self.trusted_data is None or len(updates) <= 1:
            return np.mean(updates, axis=0)
        
        X_trust, y_trust = self.trusted_data
        
        # Compute baseline update on trusted data
        pred = X_trust @ global_model.T
        error = pred - np.eye(10)[y_trust]
        baseline_grad = error.T @ X_trust / len(X_trust)
        baseline_update = -0.01 * baseline_grad
        
        # Compute similarity to baseline for each update
        similarities = []
        baseline_norm = np.linalg.norm(baseline_update)
        
        for u in updates:
            if baseline_norm > 0 and np.linalg.norm(u) > 0:
                sim = np.dot(u.flatten(), baseline_update.flatten()) / (np.linalg.norm(u) * baseline_norm)
            else:
                sim = 0.0
            similarities.append(sim)
        
        # Filter updates with low similarity
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities) if len(similarities) > 1 else 0.1
        
        trusted_updates = []
        for i, u in enumerate(updates):
            if similarities[i] > mean_sim - 2 * std_sim:
                trusted_updates.append(u)
        
        if not trusted_updates:
            trusted_updates = updates
        
        return np.mean(trusted_updates, axis=0)
    
    def _add_differential_privacy(self, update: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
        """
        Add differential privacy noise to update.
        
        Args:
            update: Aggregated update
            noise_scale: Scale of noise to add
            
        Returns:
            Update with noise added
        """
        noise = np.random.randn(*update.shape) * noise_scale * np.linalg.norm(update)
        return update + noise
    
    def defend(self, updates: List[np.ndarray], global_model: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply defense to filter and aggregate updates.
        
        Args:
            updates: List of client updates
            global_model: Current global model (needed for some defenses)
            
        Returns:
            Defended aggregated update
        """
        if not updates:
            return np.array([])
        
        filtered_updates = updates.copy()
        filtered_count = 0
        
        # Apply defense
        if self.defense_type == DefenseType.KRUM:
            aggregated = self._krum(filtered_updates, k=self.krum_k)
        elif self.defense_type == DefenseType.TRIMMED_MEAN:
            aggregated = self._trimmed_mean(filtered_updates, trim_ratio=self.trim_ratio)
        elif self.defense_type == DefenseType.ANOMALY_DETECTION:
            filtered_updates, filtered_count = self._anomaly_detection(filtered_updates, threshold=self.anomaly_threshold)
            aggregated = np.mean(filtered_updates, axis=0)
        elif self.defense_type == DefenseType.FLTRUST:
            aggregated = self._fltrust(filtered_updates, global_model)
        else:
            aggregated = np.mean(filtered_updates, axis=0)
        
        # Apply differential privacy if enabled
        if self.defense_type == DefenseType.DIFFERENTIAL_PRIVACY:
            aggregated = self._add_differential_privacy(aggregated, noise_scale=self.noise_scale)
        
        # Log defense action
        if filtered_count > 0:
            self.malicious_updates_filtered += filtered_count
            print(f"🛡️ Defense module filtered {filtered_count} malicious update(s)!")
        
        self.defense_log.append({
            'defense_type': self.defense_type,
            'updates_received': len(updates),
            'updates_filtered': filtered_count,
            'timestamp': len(self.defense_log)
        })
        
        return aggregated
    
    def get_defense_summary(self) -> Dict:
        """Get summary of defense actions."""
        return {
            'defense_type': self.defense_type,
            'total_updates_filtered': self.malicious_updates_filtered,
            'defense_actions': len(self.defense_log)
        }
