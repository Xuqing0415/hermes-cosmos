"""
Edge Device Simulator Module

Simulates edge devices with varying compute capabilities, network bandwidth,
and latency for cloud-edge federated learning.
"""

import time
import random
import threading
import numpy as np
from typing import List, Dict, Any


class SimulatedEdgeDevice:
    """
    Simulated edge device with compute, network, and storage capabilities.
    
    Attributes:
        id: Unique device identifier
        compute_speed: Relative compute speed (0.1-2.0, 1.0 = baseline)
        bandwidth: Network bandwidth in Mbps
        latency: Network latency in milliseconds
        failure_prob: Probability of failure during operations
        local_data: Local training data
        model: Current local model
        status: Device status ('online', 'offline', 'busy')
    """
    
    def __init__(self, device_id: int, compute_speed: float = 1.0, 
                 bandwidth_mbps: float = 10.0, latency_ms: int = 20,
                 failure_prob: float = 0.05):
        """
        Initialize simulated edge device.
        
        Args:
            device_id: Unique device ID
            compute_speed: Relative compute speed (0.1-2.0)
            bandwidth_mbps: Network bandwidth in Mbps
            latency_ms: Network latency in ms
            failure_prob: Probability of operation failure
        """
        self.id = device_id
        self.compute_speed = max(0.1, min(2.0, compute_speed))
        self.bandwidth = bandwidth_mbps
        self.latency = latency_ms
        self.failure_prob = failure_prob
        self.local_data = self._generate_local_data()
        self.model = None
        self.status = 'online'
        self.pending_updates = []
        
    def _generate_local_data(self) -> Dict[str, np.ndarray]:
        """Generate Non-IID local data for the device."""
        # Simulate device-specific data distribution
        np.random.seed(self.id)
        
        # Generate synthetic training data
        num_samples = int(np.random.randint(100, 500))
        input_dim = 784  # MNIST-like dimension
        num_classes = 10
        
        # Create Non-IID data by preferring certain classes
        class_prefs = np.random.choice(num_classes, size=3, replace=False)
        labels = np.random.choice(class_prefs, size=num_samples)
        
        return {
            'X': np.random.randn(num_samples, input_dim) * 0.5,
            'y': labels,
            'class_preference': class_prefs,
            'num_samples': num_samples
        }
    
    def _should_fail(self) -> bool:
        """Determine if an operation should fail randomly."""
        return random.random() < self.failure_prob
    
    def train(self, global_model: np.ndarray, num_epochs: int = 1, 
              batch_size: int = 32) -> np.ndarray:
        """
        Simulate local training on the device.
        
        Args:
            global_model: Global model parameters
            num_epochs: Number of local training epochs
            batch_size: Training batch size
        
        Returns:
            Model update (delta from global model)
        """
        if self.status != 'online':
            raise RuntimeError(f"Device {self.id} is not online")
        
        if self._should_fail():
            self.status = 'offline'
            raise RuntimeError(f"Device {self.id} failed during training")
        
        self.status = 'busy'
        
        try:
            # Simulate training time based on compute speed and data size
            train_time = (0.3 / self.compute_speed) * num_epochs * (len(self.local_data['y']) / 200)
            time.sleep(train_time)
            
            # Simulate gradient computation
            update = self._compute_update(global_model)
            
            return update
        finally:
            self.status = 'online'
    
    def _compute_update(self, global_model: np.ndarray) -> np.ndarray:
        """Compute model update based on local data."""
        # Simulate gradient descent update
        # The update depends on local data characteristics
        gradient = np.random.randn(*global_model.shape) * 0.01
        
        # Add device-specific bias based on compute speed
        # Slower devices have noisier gradients
        noise_scale = 1.0 + (1.0 - self.compute_speed) * 0.5
        gradient *= noise_scale
        
        return gradient
    
    def download_model(self, model: np.ndarray) -> bool:
        """
        Simulate downloading model from cloud.
        
        Args:
            model: Model parameters to download
        
        Returns:
            True if successful, False if failed
        """
        if self._should_fail():
            self.status = 'offline'
            return False
        
        # Simulate download delay
        data_size_mb = model.nbytes / (1024 * 1024)
        transfer_time = (data_size_mb * 8) / self.bandwidth  # Convert MB to Mbps
        delay = transfer_time + (self.latency / 1000)
        time.sleep(delay)
        
        self.model = model.copy()
        return True
    
    def upload_update(self, update: np.ndarray) -> bool:
        """
        Simulate uploading update to cloud.
        
        Args:
            update: Model update to upload
        
        Returns:
            True if successful, False if failed
        """
        if self._should_fail():
            self.status = 'offline'
            return False
        
        # Simulate upload delay
        data_size_mb = update.nbytes / (1024 * 1024)
        transfer_time = (data_size_mb * 8) / self.bandwidth
        delay = transfer_time + (self.latency / 1000)
        time.sleep(delay)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get device statistics."""
        return {
            'device_id': self.id,
            'compute_speed': self.compute_speed,
            'bandwidth': self.bandwidth,
            'latency': self.latency,
            'status': self.status,
            'data_size': len(self.local_data['y']),
            'class_preference': self.local_data['class_preference'].tolist()
        }
    
    def reset(self):
        """Reset device status to online."""
        self.status = 'online'


class EdgeDeviceFactory:
    """Factory class for creating edge devices with realistic distributions."""
    
    @staticmethod
    def create_device(device_id: int, profile: str = 'mixed') -> SimulatedEdgeDevice:
        """
        Create a device with specific profile.
        
        Args:
            device_id: Device ID
            profile: Device profile ('mobile', 'tablet', 'desktop', 'iot', 'mixed')
        
        Returns:
            Simulated edge device
        """
        profiles = {
            'mobile': {
                'compute_speed': (0.3, 0.7),
                'bandwidth': (5, 20),
                'latency': (50, 200),
                'failure_prob': 0.1
            },
            'tablet': {
                'compute_speed': (0.6, 1.0),
                'bandwidth': (10, 50),
                'latency': (30, 100),
                'failure_prob': 0.05
            },
            'desktop': {
                'compute_speed': (1.0, 1.5),
                'bandwidth': (50, 100),
                'latency': (10, 30),
                'failure_prob': 0.02
            },
            'iot': {
                'compute_speed': (0.1, 0.4),
                'bandwidth': (1, 10),
                'latency': (100, 500),
                'failure_prob': 0.15
            }
        }
        
        if profile == 'mixed':
            profile = random.choice(list(profiles.keys()))
        
        p = profiles[profile]
        return SimulatedEdgeDevice(
            device_id=device_id,
            compute_speed=random.uniform(*p['compute_speed']),
            bandwidth_mbps=random.uniform(*p['bandwidth']),
            latency_ms=int(random.uniform(*p['latency'])),
            failure_prob=p['failure_prob']
        )
    
    @staticmethod
    def create_cluster(num_devices: int, distribution: str = 'heterogeneous') -> List[SimulatedEdgeDevice]:
        """
        Create a cluster of edge devices.
        
        Args:
            num_devices: Number of devices to create
            distribution: 'homogeneous' or 'heterogeneous'
        
        Returns:
            List of edge devices
        """
        devices = []
        
        if distribution == 'homogeneous':
            # All devices have similar characteristics
            base_speed = random.uniform(0.8, 1.2)
            base_bandwidth = random.uniform(20, 50)
            base_latency = int(random.uniform(20, 50))
            
            for i in range(num_devices):
                devices.append(SimulatedEdgeDevice(
                    device_id=i,
                    compute_speed=base_speed * random.uniform(0.9, 1.1),
                    bandwidth_mbps=base_bandwidth * random.uniform(0.9, 1.1),
                    latency_ms=int(base_latency * random.uniform(0.9, 1.1))
                ))
        else:
            # Mixed device types
            for i in range(num_devices):
                devices.append(EdgeDeviceFactory.create_device(i, profile='mixed'))
        
        return devices


# Example usage
if __name__ == "__main__":
    print("=== Testing Edge Device Simulator ===")
    
    # Create devices with different profiles
    devices = [
        EdgeDeviceFactory.create_device(0, 'mobile'),
        EdgeDeviceFactory.create_device(1, 'desktop'),
        EdgeDeviceFactory.create_device(2, 'iot')
    ]
    
    for device in devices:
        stats = device.get_stats()
        print(f"\nDevice {stats['device_id']}:")
        print(f"  Type: {stats['status']}")
        print(f"  Compute Speed: {stats['compute_speed']:.2f}x")
        print(f"  Bandwidth: {stats['bandwidth']:.1f} Mbps")
        print(f"  Latency: {stats['latency']} ms")
        print(f"  Data Size: {stats['data_size']} samples")
    
    # Test training
    test_model = np.random.randn(10, 784)
    device = devices[0]
    
    try:
        update = device.train(test_model, num_epochs=2)
        print(f"\nTraining successful! Update shape: {update.shape}")
    except Exception as e:
        print(f"\nTraining failed: {e}")
