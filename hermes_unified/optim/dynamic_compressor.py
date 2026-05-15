"""
Dynamic Gradient Compressor

Implements adaptive gradient compression with error compensation.
Supports Top-k and 1-bit compression schemes.
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, Optional


class DynamicCompressor:
    """
    Dynamic gradient compressor with adaptive compression ratio.
    
    Key features:
    - Supports Top-k and 1-bit compression
    - Maintains error compensation buffer to prevent convergence drift
    - Adjusts compression ratio based on measured network bandwidth
    - Uses proportional control for smooth ratio transitions
    """
    
    def __init__(self, 
                 initial_ratio: float = 0.1, 
                 min_ratio: float = 0.01, 
                 max_ratio: float = 1.0,
                 compression_type: str = 'topk'):
        """
        Initialize dynamic compressor.
        
        Args:
            initial_ratio: Initial compression ratio (0-1)
            min_ratio: Minimum allowed compression ratio
            max_ratio: Maximum allowed compression ratio
            compression_type: 'topk' or '1bit'
        """
        self.comp_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.compression_type = compression_type.lower()
        
        # Error compensation buffer
        self.error_buffer: Dict[int, np.ndarray] = {}
        
        # Bandwidth history for adaptive adjustment
        self.bandwidth_history = deque(maxlen=10)
        self.target_bandwidth = 100.0  # MB/s
        
    def update_congestion(self, measured_bandwidth: float):
        """
        Update compression ratio based on measured bandwidth.
        
        Args:
            measured_bandwidth: Current bandwidth in MB/s
        """
        self.bandwidth_history.append(measured_bandwidth)
        
        if len(self.bandwidth_history) < 3:
            return
        
        avg_bandwidth = np.mean(list(self.bandwidth_history))
        
        # Proportional control: adjust based on bandwidth utilization
        target_utilization = 0.7
        current_utilization = avg_bandwidth / self.target_bandwidth
        
        if current_utilization > 1.0:
            # Congested: increase compression
            self.comp_ratio = max(self.min_ratio, self.comp_ratio * 0.9)
        elif current_utilization < target_utilization:
            # Underutilized: decrease compression
            self.comp_ratio = min(self.max_ratio, self.comp_ratio * 1.05)
    
    def compress(self, gradient: np.ndarray, tensor_id: int = 0) -> Tuple[Tuple, Dict]:
        """
        Compress gradient with error compensation.
        
        Args:
            gradient: Input gradient tensor
            tensor_id: Unique identifier for this tensor (for error tracking)
            
        Returns:
            Tuple of (compressed_data, metadata)
        """
        # Add accumulated error from previous iterations
        if tensor_id in self.error_buffer:
            gradient = gradient + self.error_buffer[tensor_id]
        
        if self.compression_type == 'topk':
            return self._compress_topk(gradient, tensor_id)
        elif self.compression_type == '1bit':
            return self._compress_1bit(gradient)
        else:
            return gradient, {'ratio': 1.0, 'method': 'full'}
    
    def _compress_topk(self, gradient: np.ndarray, tensor_id: int) -> Tuple[Tuple, Dict]:
        """Top-k compression with error compensation."""
        if self.comp_ratio >= 1.0:
            return gradient, {'ratio': 1.0, 'method': 'topk', 'shape': gradient.shape}
        
        flat_grad = gradient.flatten()
        n_elements = flat_grad.size
        n_top = max(1, int(n_elements * self.comp_ratio))
        
        # Get top-k indices and values
        top_indices = np.argpartition(np.abs(flat_grad), -n_top)[-n_top:]
        top_values = flat_grad[top_indices]
        
        # Store error for compensation
        error = flat_grad.copy()
        error[top_indices] = 0.0
        
        if tensor_id in self.error_buffer:
            self.error_buffer[tensor_id] = self.error_buffer[tensor_id] * 0.9 + error * 0.1
        else:
            self.error_buffer[tensor_id] = error
        
        return (top_indices, top_values), {
            'ratio': self.comp_ratio,
            'method': 'topk',
            'shape': gradient.shape,
            'n_top': n_top
        }
    
    def _compress_1bit(self, gradient: np.ndarray) -> Tuple[Tuple, Dict]:
        """1-bit compression (sign-based)."""
        sign = np.sign(gradient)
        magnitude = np.mean(np.abs(gradient))
        
        return (sign, magnitude), {
            'ratio': 1.0 / gradient.size,
            'method': '1bit',
            'shape': gradient.shape
        }
    
    def decompress(self, compressed_data: Tuple, metadata: Dict) -> np.ndarray:
        """
        Decompress gradient.
        
        Args:
            compressed_data: Compressed data tuple
            metadata: Compression metadata
            
        Returns:
            Decompressed gradient tensor
        """
        if metadata['method'] == 'topk':
            return self._decompress_topk(compressed_data, metadata)
        elif metadata['method'] == '1bit':
            return self._decompress_1bit(compressed_data, metadata)
        else:
            return compressed_data
    
    def _decompress_topk(self, compressed_data: Tuple, metadata: Dict) -> np.ndarray:
        """Decompress Top-k gradient."""
        indices, values = compressed_data
        shape = metadata['shape']
        
        decompressed = np.zeros(int(np.prod(shape)))
        decompressed[indices] = values
        
        return decompressed.reshape(shape)
    
    def _decompress_1bit(self, compressed_data: Tuple, metadata: Dict) -> np.ndarray:
        """Decompress 1-bit gradient."""
        sign, magnitude = compressed_data
        return sign * magnitude
    
    def reset_error(self):
        """Reset error compensation buffer."""
        self.error_buffer.clear()
    
    def get_stats(self) -> Dict:
        """Get compression statistics."""
        return {
            'compression_ratio': self.comp_ratio,
            'compression_type': self.compression_type,
            'error_buffers': len(self.error_buffer),
            'avg_bandwidth': np.mean(list(self.bandwidth_history)) if self.bandwidth_history else 0.0
        }


class CompressionAnalyzer:
    """
    Analyzer for evaluating compression effectiveness.
    
    Measures:
    - Compression ratio achieved
    - Error introduced by compression
    - Impact on convergence
    """
    
    def __init__(self):
        self.original_norms = []
        self.compressed_norms = []
        self.error_norms = []
    
    def record_compression(self, original: np.ndarray, compressed: np.ndarray):
        """Record compression statistics."""
        original_norm = np.linalg.norm(original)
        compressed_norm = np.linalg.norm(compressed)
        error_norm = np.linalg.norm(original - compressed)
        
        self.original_norms.append(original_norm)
        self.compressed_norms.append(compressed_norm)
        self.error_norms.append(error_norm)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.original_norms:
            return {}
        
        return {
            'avg_original_norm': np.mean(self.original_norms),
            'avg_compressed_norm': np.mean(self.compressed_norms),
            'avg_error_norm': np.mean(self.error_norms),
            'avg_error_ratio': np.mean([e / o for e, o in zip(self.error_norms, self.original_norms)]),
            'sample_count': len(self.original_norms)
        }
