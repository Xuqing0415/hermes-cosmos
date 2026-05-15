"""
Model Pruning Module

Implements adaptive model pruning based on device capabilities.
"""

import numpy as np
from typing import Dict, Any


def prune_model_by_speed(model: np.ndarray, speed: float) -> np.ndarray:
    """
    Prune model based on device compute speed.
    
    Args:
        model: Original model parameters
        speed: Device compute speed (0.1-2.0)
    
    Returns:
        Pruned model
    """
    if speed > 0.8:
        # High speed device: no pruning
        return model.copy()
    elif speed > 0.5:
        # Medium speed device: moderate pruning
        return prune_by_ratio(model, 0.7)
    else:
        # Low speed device: aggressive pruning
        return prune_by_ratio(model, 0.4)


def prune_by_ratio(model: np.ndarray, ratio: float) -> np.ndarray:
    """
    Prune model by keeping a specified ratio of channels/neurons.
    
    Args:
        model: Original model parameters
        ratio: Fraction of parameters to keep (0.0-1.0)
    
    Returns:
        Pruned model
    """
    pruned = model.copy()
    
    if model.ndim == 2:
        # Linear layer: prune columns (output features)
        num_features = model.shape[1]
        num_to_keep = max(1, int(num_features * ratio))
        
        # Select most important features based on column norms
        norms = np.linalg.norm(model, axis=0)
        keep_indices = np.argsort(norms)[-num_to_keep:]
        
        pruned = pruned[:, keep_indices]
    
    elif model.ndim == 4:
        # Convolutional layer: prune output channels
        num_channels = model.shape[0]
        num_to_keep = max(1, int(num_channels * ratio))
        
        norms = np.linalg.norm(model.reshape(num_channels, -1), axis=1)
        keep_indices = np.argsort(norms)[-num_to_keep:]
        
        pruned = pruned[keep_indices, :, :, :]
    
    else:
        # For other shapes, just scale down
        pruned = pruned * ratio
    
    return pruned


def quantize_model(model: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    Quantize model to reduce size.
    
    Args:
        model: Original model parameters
        bits: Number of bits for quantization (8 or 4)
    
    Returns:
        Quantized model
    """
    if bits == 8:
        # INT8 quantization
        min_val = np.min(model)
        max_val = np.max(model)
        scale = (max_val - min_val) / 255.0
        quantized = np.round((model - min_val) / scale).astype(np.int8)
        return quantized
    elif bits == 4:
        # INT4 quantization
        min_val = np.min(model)
        max_val = np.max(model)
        scale = (max_val - min_val) / 15.0
        quantized = np.round((model - min_val) / scale).astype(np.int8)  # Use int8 for storage
        return quantized
    else:
        return model.copy()


def dequantize_model(quantized_model: np.ndarray, original_min: float, original_max: float, bits: int = 8) -> np.ndarray:
    """
    Dequantize model back to float.
    
    Args:
        quantized_model: Quantized model
        original_min: Original minimum value
        original_max: Original maximum value
        bits: Number of bits used for quantization
    
    Returns:
        Dequantized float model
    """
    if bits == 8:
        scale = (original_max - original_min) / 255.0
        return quantized_model.astype(np.float32) * scale + original_min
    elif bits == 4:
        scale = (original_max - original_min) / 15.0
        return quantized_model.astype(np.float32) * scale + original_min
    else:
        return quantized_model.astype(np.float32)


def get_model_size(model: np.ndarray) -> float:
    """Get model size in MB."""
    return model.nbytes / (1024 * 1024)


class AdaptiveModelManager:
    """
    Manages adaptive model compression for edge devices.
    
    Attributes:
        original_model: Full precision original model
        pruning_cache: Cache of pruned models
        quantization_cache: Cache of quantized models
    """
    
    def __init__(self, model: np.ndarray):
        """
        Initialize adaptive model manager.
        
        Args:
            model: Full precision model
        """
        self.original_model = model.copy()
        self.pruning_cache: Dict[float, np.ndarray] = {}
        self.quantization_cache: Dict[tuple, np.ndarray] = {}
        
        # Store original stats for dequantization
        self.original_min = np.min(model)
        self.original_max = np.max(model)
    
    def get_adapted_model(self, compute_speed: float, bandwidth: float) -> Dict[str, Any]:
        """
        Get model adapted to device capabilities.
        
        Args:
            compute_speed: Device compute speed
            bandwidth: Device bandwidth in Mbps
        
        Returns:
            Dictionary with adapted model and metadata
        """
        # First apply pruning based on compute speed
        if compute_speed not in self.pruning_cache:
            self.pruning_cache[compute_speed] = prune_model_by_speed(self.original_model, compute_speed)
        
        pruned_model = self.pruning_cache[compute_speed]
        
        # Then apply quantization based on bandwidth
        # Lower bandwidth = more aggressive quantization
        if bandwidth < 5:
            bits = 4
        elif bandwidth < 20:
            bits = 8
        else:
            bits = 32  # No quantization
        
        if bits < 32:
            cache_key = (compute_speed, bits)
            if cache_key not in self.quantization_cache:
                self.quantization_cache[cache_key] = quantize_model(pruned_model, bits)
            
            return {
                'model': self.quantization_cache[cache_key],
                'type': 'quantized',
                'bits': bits,
                'pruning_ratio': self._get_pruning_ratio(compute_speed),
                'original_min': self.original_min,
                'original_max': self.original_max,
                'size_mb': get_model_size(self.quantization_cache[cache_key])
            }
        else:
            return {
                'model': pruned_model,
                'type': 'pruned',
                'bits': 32,
                'pruning_ratio': self._get_pruning_ratio(compute_speed),
                'size_mb': get_model_size(pruned_model)
            }
    
    def _get_pruning_ratio(self, speed: float) -> float:
        """Get pruning ratio for a given speed."""
        if speed > 0.8:
            return 1.0
        elif speed > 0.5:
            return 0.7
        else:
            return 0.4
    
    def reconstruct_update(self, device_update: np.ndarray, device_info: Dict[str, Any]) -> np.ndarray:
        """
        Reconstruct full update from device update.
        
        Args:
            device_update: Update from device (possibly quantized/pruned)
            device_info: Device metadata (bits, pruning_ratio, etc.)
        
        Returns:
            Full-size update matching original model shape
        """
        # Dequantize if needed
        if device_info['type'] == 'quantized':
            update = dequantize_model(
                device_update,
                device_info['original_min'],
                device_info['original_max'],
                device_info['bits']
            )
        else:
            update = device_update
        
        # Expand to original size if pruned
        original_shape = self.original_model.shape
        if update.shape != original_shape:
            expanded = np.zeros(original_shape, dtype=update.dtype)
            
            if update.ndim == 2 and original_shape[0] == update.shape[0]:
                # Linear layer: restore columns
                num_cols = update.shape[1]
                expanded[:, :num_cols] = update
            elif update.ndim == 4 and original_shape[1:] == update.shape[1:]:
                # Conv layer: restore channels
                num_chans = update.shape[0]
                expanded[:num_chans, :, :, :] = update
            else:
                expanded[:update.shape[0], :update.shape[1]] = update
            
            return expanded
        
        return update


# Example usage
if __name__ == "__main__":
    print("=== Testing Pruning Module ===")
    
    # Create test model
    test_model = np.random.randn(10, 784)
    print(f"Original model shape: {test_model.shape}")
    print(f"Original model size: {get_model_size(test_model):.4f} MB")
    
    # Test pruning
    pruned = prune_by_ratio(test_model, 0.5)
    print(f"\nPruned model shape: {pruned.shape}")
    print(f"Pruned model size: {get_model_size(pruned):.4f} MB")
    
    # Test adaptive pruning
    manager = AdaptiveModelManager(test_model)
    
    for speed in [0.3, 0.6, 1.0]:
        adapted = manager.get_adapted_model(speed, bandwidth=10)
        print(f"\nSpeed {speed}:")
        print(f"  Type: {adapted['type']}")
        print(f"  Bits: {adapted['bits']}")
        print(f"  Size: {adapted['size_mb']:.4f} MB")
        print(f"  Shape: {adapted['model'].shape}")
