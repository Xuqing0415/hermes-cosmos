"""
Drift Detection for Federated Online Learning

Implements various drift detection algorithms for streaming data.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque


class DriftDetector:
    """Base class for drift detection algorithms."""
    
    def __init__(self):
        self.drift_detected = False
        self.warnings = []
    
    def update(self, value: float) -> bool:
        """
        Update detector with new value.
        
        Args:
            value: New metric value (e.g., loss, accuracy)
        
        Returns:
            True if drift detected
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset detector state."""
        self.drift_detected = False
        self.warnings = []
    
    def get_state(self) -> Dict[str, Any]:
        """Get detector state."""
        return {
            'drift_detected': self.drift_detected,
            'warnings': self.warnings
        }


class CUSUMDetector(DriftDetector):
    """
    Cumulative Sum (CUSUM) drift detector.
    
    Detects changes in the mean of a process by accumulating deviations
    from the target mean.
    """
    
    def __init__(self, target_mean: float = 0.0, delta: float = 0.05, 
                 threshold: float = 5.0, reset_after_drift: bool = True):
        super().__init__()
        
        self.target_mean = target_mean
        self.delta = delta
        self.threshold = threshold
        self.reset_after_drift = reset_after_drift
        
        self.cumulative_sum = 0.0
        self.count = 0
    
    def update(self, value: float) -> bool:
        """Update CUSUM detector."""
        self.count += 1
        
        deviation = value - self.target_mean - self.delta
        self.cumulative_sum = max(0.0, self.cumulative_sum + deviation)
        
        if self.cumulative_sum > self.threshold:
            self.drift_detected = True
            self.warnings.append(f"Drift detected at sample {self.count}")
            
            if self.reset_after_drift:
                self.reset()
            
            return True
        
        return False
    
    def reset(self):
        """Reset CUSUM state."""
        super().reset()
        self.cumulative_sum = 0.0


class ADWINDetector(DriftDetector):
    """
    Adaptive Windowing (ADWIN) drift detector.
    
    Maintains a sliding window that adapts its size based on detecting
    concept drift.
    """
    
    def __init__(self, delta: float = 0.001, max_buckets: int = 5,
                 min_window_size: int = 10):
        super().__init__()
        
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window_size = min_window_size
        
        self.buckets = []
        self.total_sum = 0.0
        self.total_count = 0
    
    def update(self, value: float) -> bool:
        """Update ADWIN detector."""
        self._add_element(value)
        
        if self._detect_drift():
            self.drift_detected = True
            self.warnings.append(f"Drift detected at sample {self.total_count}")
            self._reduce_window()
            return True
        
        return False
    
    def _add_element(self, value: float):
        """Add element to bucket structure."""
        if self.buckets and len(self.buckets[-1]) < self.max_buckets:
            self.buckets[-1].append(value)
        else:
            self.buckets.append([value])
        
        self.total_sum += value
        self.total_count += 1
        
        self._compress_buckets()
    
    def _compress_buckets(self):
        """Compress buckets to maintain efficiency."""
        i = 0
        while i < len(self.buckets) - 1:
            if len(self.buckets[i]) == len(self.buckets[i+1]) == self.max_buckets:
                merged = self.buckets[i] + self.buckets[i+1]
                self.buckets[i] = merged[:self.max_buckets]
                self.buckets[i+1] = merged[self.max_buckets:]
                if not self.buckets[i+1]:
                    del self.buckets[i+1]
            i += 1
    
    def _detect_drift(self) -> bool:
        """Detect drift using ADWIN's statistical test."""
        if self.total_count < self.min_window_size:
            return False
        
        for i in range(1, len(self.buckets)):
            mean_left = self._compute_mean(range(i))
            mean_right = self._compute_mean(range(i, len(self.buckets)))
            
            if abs(mean_left - mean_right) > self._compute_bound(i):
                return True
        
        return False
    
    def _compute_mean(self, bucket_indices: range) -> float:
        """Compute mean over specified buckets."""
        total = 0.0
        count = 0
        
        for i in bucket_indices:
            total += sum(self.buckets[i])
            count += len(self.buckets[i])
        
        return total / count if count > 0 else 0.0
    
    def _compute_bound(self, split_point: int) -> float:
        """Compute statistical bound for drift detection."""
        n = self.total_count
        n0 = sum(len(b) for b in self.buckets[:split_point])
        n1 = n - n0
        
        if n0 == 0 or n1 == 0:
            return float('inf')
        
        log_term = np.log(3 * n / self.delta)
        bound = np.sqrt((log_term) / (2 * min(n0, n1)))
        
        return bound
    
    def _reduce_window(self):
        """Reduce window after drift detection."""
        if len(self.buckets) > 1:
            self.buckets = self.buckets[1:]
            self.total_sum = sum(sum(b) for b in self.buckets)
            self.total_count = sum(len(b) for b in self.buckets)
    
    def reset(self):
        """Reset ADWIN state."""
        super().reset()
        self.buckets = []
        self.total_sum = 0.0
        self.total_count = 0


class PageHinkleyDetector(DriftDetector):
    """
    Page-Hinkley drift detector.
    
    Detects changes in the mean by tracking cumulative sums of deviations
    with a threshold.
    """
    
    def __init__(self, delta: float = 0.005, lambda_: float = 50.0,
                 alpha: float = 0.99, min_instances: int = 30):
        super().__init__()
        
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.min_instances = min_instances
        
        self.x_mean = 0.0
        self.sum_ = 0.0
        self.sum_min = 0.0
        self.count = 0
    
    def update(self, value: float) -> bool:
        """Update Page-Hinkley detector."""
        self.count += 1
        
        self.x_mean = (self.x_mean * (self.count - 1) + value) / self.count
        
        self.sum_ = self.alpha * self.sum_ + (value - self.x_mean - self.delta)
        self.sum_min = min(self.sum_min, self.sum_)
        
        if self.count > self.min_instances:
            if self.sum_ - self.sum_min > self.lambda_:
                self.drift_detected = True
                self.warnings.append(f"Drift detected at sample {self.count}")
                return True
        
        return False
    
    def reset(self):
        """Reset Page-Hinkley state."""
        super().reset()
        self.x_mean = 0.0
        self.sum_ = 0.0
        self.sum_min = 0.0
        self.count = 0


class WindowedDriftDetector(DriftDetector):
    """
    Simple windowed drift detector.
    
    Compares recent window statistics with historical statistics.
    """
    
    def __init__(self, window_size: int = 100, threshold_ratio: float = 2.0):
        super().__init__()
        
        self.window_size = window_size
        self.threshold_ratio = threshold_ratio
        
        self.window = deque(maxlen=window_size)
        self.historical_mean = 0.0
        self.historical_std = 1.0
        self.count = 0
    
    def update(self, value: float) -> bool:
        """Update windowed detector."""
        self.window.append(value)
        self.count += 1
        
        if len(self.window) >= self.window_size:
            recent_mean = np.mean(self.window)
            recent_std = np.std(self.window)
            
            if self.count >= 2 * self.window_size:
                ratio = abs(recent_mean - self.historical_mean) / max(self.historical_std, recent_std, 1e-6)
                
                if ratio > self.threshold_ratio:
                    self.drift_detected = True
                    self.warnings.append(f"Drift detected at sample {self.count}")
                    return True
            
            self.historical_mean = (self.historical_mean * (self.count - 1) + value) / self.count
            self.historical_std = np.sqrt(
                ((self.count - 1) * self.historical_std ** 2 + 
                 (value - self.historical_mean) ** 2) / self.count
            )
        
        return False
    
    def reset(self):
        """Reset windowed detector state."""
        super().reset()
        self.window.clear()
        self.historical_mean = 0.0
        self.historical_std = 1.0
        self.count = 0


class DriftDetectorEnsemble(DriftDetector):
    """
    Ensemble of multiple drift detectors.
    
    Combines multiple detection methods for improved robustness.
    """
    
    def __init__(self, detectors: List[DriftDetector], threshold: float = 0.5):
        super().__init__()
        
        self.detectors = detectors
        self.threshold = threshold
    
    def update(self, value: float) -> bool:
        """Update all detectors and combine results."""
        votes = []
        
        for detector in self.detectors:
            detected = detector.update(value)
            votes.append(detected)
        
        detection_ratio = sum(votes) / len(self.detectors)
        
        if detection_ratio >= self.threshold:
            self.drift_detected = True
            self.warnings.append(f"Ensemble drift detected (ratio={detection_ratio:.2f})")
            return True
        
        return False
    
    def reset(self):
        """Reset all detectors."""
        super().reset()
        for detector in self.detectors:
            detector.reset()
    
    def get_state(self) -> Dict[str, Any]:
        """Get detailed state of all detectors."""
        states = {}
        for i, detector in enumerate(self.detectors):
            states[f'detector_{i}'] = detector.get_state()
        
        states['ensemble_drift'] = self.drift_detected
        return states


class ConceptDriftSimulator:
    """Simulates concept drift in streaming data."""
    
    def __init__(self, drift_type: str = 'sudden', drift_interval: int = 1000):
        self.drift_type = drift_type
        self.drift_interval = drift_interval
        self.count = 0
        self.drift_active = False
    
    def should_drift(self) -> bool:
        """Check if it's time for drift."""
        self.count += 1
        
        if self.drift_type == 'sudden':
            if self.count % self.drift_interval == 0 and not self.drift_active:
                self.drift_active = True
                return True
            elif self.count % self.drift_interval == 100:
                self.drift_active = False
            
        elif self.drift_type == 'gradual':
            if (self.count % self.drift_interval) < 100:
                self.drift_active = True
                return True
        
        elif self.drift_type == 'recurring':
            cycle = self.drift_interval
            phase = self.count % cycle
            if phase > cycle * 0.8:
                self.drift_active = True
                return True
        
        return False
    
    def apply_drift(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply drift to data.
        
        Args:
            X: Feature matrix
            y: Labels
        
        Returns:
            Modified X, y with drift applied
        """
        if not self.drift_active:
            return X, y
        
        if self.drift_type == 'sudden':
            y = 1 - y
        
        elif self.drift_type == 'gradual':
            noise = np.random.randn(len(y)) * 0.1
            y = np.clip(y + noise, 0, 1)
        
        elif self.drift_type == 'recurring':
            y = np.where(np.random.rand(len(y)) > 0.5, y, 1 - y)
        
        return X, y