"""
Cloud Server Module

Implements the cloud-side server for managing cloud-edge federated learning.
"""

import numpy as np
import time
import threading
from typing import List, Dict, Any, Optional
from .edge_simulator import SimulatedEdgeDevice


class CloudServer:
    """
    Cloud server for managing federated learning across edge devices.
    
    Attributes:
        global_model: Current global model parameters
        devices: Registered edge devices
        aggregation_method: Method for aggregating updates ('fedavg', 'weighted')
        round_history: History of round results
        total_comm_cost: Total communication cost in MB
    """
    
    def __init__(self, global_model: np.ndarray, aggregation_method: str = 'fedavg'):
        """
        Initialize cloud server.
        
        Args:
            global_model: Initial global model parameters
            aggregation_method: Aggregation method
        """
        self.global_model = global_model.copy()
        self.devices: List[SimulatedEdgeDevice] = []
        self.aggregation_method = aggregation_method
        self.round_history = []
        self.total_comm_cost = 0.0
        self.current_round = 0
    
    def register_device(self, device: SimulatedEdgeDevice):
        """Register an edge device with the server."""
        self.devices.append(device)
        print(f"Device {device.id} registered")
    
    def select_devices(self, num_devices: int, strategy: str = 'random') -> List[SimulatedEdgeDevice]:
        """
        Select devices for the next round.
        
        Args:
            num_devices: Number of devices to select
            strategy: Selection strategy ('random', 'fastest', 'reliable')
        
        Returns:
            List of selected devices
        """
        online_devices = [d for d in self.devices if d.status == 'online']
        
        if len(online_devices) == 0:
            return []
        
        if strategy == 'random':
            return np.random.choice(online_devices, min(num_devices, len(online_devices)), replace=False).tolist()
        
        elif strategy == 'fastest':
            # Select devices with highest compute speed
            sorted_devices = sorted(online_devices, key=lambda d: d.compute_speed, reverse=True)
            return sorted_devices[:num_devices]
        
        elif strategy == 'reliable':
            # Select devices with lowest failure probability
            sorted_devices = sorted(online_devices, key=lambda d: d.failure_prob)
            return sorted_devices[:num_devices]
        
        elif strategy == 'balanced':
            # Balance between speed and reliability
            sorted_devices = sorted(online_devices, key=lambda d: d.compute_speed / (d.failure_prob + 0.01), reverse=True)
            return sorted_devices[:num_devices]
        
        else:
            return np.random.choice(online_devices, min(num_devices, len(online_devices)), replace=False).tolist()
    
    def round(self, selected_devices: List[SimulatedEdgeDevice], num_epochs: int = 1, 
              max_retries: int = 2, timeout: float = 30.0) -> int:
        """
        Execute one round of federated learning.
        
        Args:
            selected_devices: List of devices to participate
            num_epochs: Number of local epochs
            max_retries: Maximum retries for failed devices
            timeout: Maximum time per device in seconds
        
        Returns:
            Number of successful updates received
        """
        self.current_round += 1
        updates = []
        successful_devices = []
        failed_devices = []
        
        for device in selected_devices:
            success = False
            retries = 0
            
            while retries < max_retries and not success:
                try:
                    # Download model
                    if not device.download_model(self.global_model):
                        raise RuntimeError("Download failed")
                    
                    # Local training
                    update = device.train(self.global_model, num_epochs=num_epochs)
                    
                    # Upload update
                    if not device.upload_update(update):
                        raise RuntimeError("Upload failed")
                    
                    updates.append((update, device.compute_speed))
                    successful_devices.append(device.id)
                    success = True
                    
                except Exception as e:
                    retries += 1
                    device.reset()
                    time.sleep(0.1)
                    if retries >= max_retries:
                        failed_devices.append(device.id)
        
        # Aggregate updates
        if updates:
            self._aggregate(updates)
        
        # Record round statistics
        round_stats = {
            'round': self.current_round,
            'selected': len(selected_devices),
            'successful': len(successful_devices),
            'failed': len(failed_devices),
            'successful_devices': successful_devices,
            'failed_devices': failed_devices,
            'model_norm': float(np.linalg.norm(self.global_model))
        }
        self.round_history.append(round_stats)
        
        return len(successful_devices)
    
    def _aggregate(self, updates: List[tuple]):
        """Aggregate updates from devices."""
        if self.aggregation_method == 'fedavg':
            # Simple average
            update_arrays = [u[0] for u in updates]
            avg_update = np.mean(update_arrays, axis=0)
            self.global_model += avg_update
            
        elif self.aggregation_method == 'weighted':
            # Weighted by compute speed
            total_weight = sum(u[1] for u in updates)
            weighted_updates = [u[0] * (u[1] / total_weight) for u in updates]
            avg_update = np.sum(weighted_updates, axis=0)
            self.global_model += avg_update
        
        # Update communication cost
        for update, _ in updates:
            self.total_comm_cost += update.nbytes / (1024 * 1024) * 2  # Download + Upload
    
    def async_round(self, selected_devices: List[SimulatedEdgeDevice], num_epochs: int = 1):
        """
        Execute an asynchronous round of federated learning.
        
        Args:
            selected_devices: List of devices to participate
            num_epochs: Number of local epochs
        """
        results = []
        
        def async_train(device):
            try:
                device.download_model(self.global_model)
                update = device.train(self.global_model, num_epochs=num_epochs)
                device.upload_update(update)
                results.append((update, device.compute_speed))
            except Exception as e:
                print(f"Device {device.id} failed in async round: {e}")
        
        threads = []
        for device in selected_devices:
            t = threading.Thread(target=async_train, args=(device,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=60)
        
        # Aggregate completed updates
        if results:
            self._aggregate(results)
        
        return len(results)
    
    def evaluate(self, test_data: Dict[str, np.ndarray]) -> float:
        """
        Evaluate the global model on test data.
        
        Args:
            test_data: Dictionary with 'X' and 'y' keys
        
        Returns:
            Accuracy score
        """
        # Simple linear classifier evaluation (simulated)
        predictions = np.argmax(test_data['X'] @ self.global_model.T, axis=1)
        accuracy = np.mean(predictions == test_data['y'])
        return accuracy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        online_count = sum(1 for d in self.devices if d.status == 'online')
        avg_compute_speed = np.mean([d.compute_speed for d in self.devices])
        avg_bandwidth = np.mean([d.bandwidth for d in self.devices])
        avg_latency = np.mean([d.latency for d in self.devices])
        
        return {
            'num_devices': len(self.devices),
            'online_devices': online_count,
            'avg_compute_speed': avg_compute_speed,
            'avg_bandwidth': avg_bandwidth,
            'avg_latency': avg_latency,
            'current_round': self.current_round,
            'total_comm_cost': self.total_comm_cost,
            'model_size_mb': self.global_model.nbytes / (1024 * 1024)
        }
    
    def get_round_history(self) -> List[Dict]:
        """Get history of all rounds."""
        return self.round_history


class CloudEdgeCoordinator:
    """
    Coordinator for cloud-edge federated learning.
    
    Manages the overall training process and provides high-level APIs.
    """
    
    def __init__(self, model_shape: tuple = (10, 784), num_devices: int = 20, 
                 device_distribution: str = 'heterogeneous'):
        """
        Initialize coordinator.
        
        Args:
            model_shape: Shape of the model parameters
            num_devices: Number of edge devices
            device_distribution: 'homogeneous' or 'heterogeneous'
        """
        from .edge_simulator import EdgeDeviceFactory
        
        self.global_model = np.random.randn(*model_shape) * 0.01
        self.server = CloudServer(self.global_model, aggregation_method='fedavg')
        self.devices = EdgeDeviceFactory.create_cluster(num_devices, distribution=device_distribution)
        
        for device in self.devices:
            self.server.register_device(device)
        
        # Create test data
        self.test_data = self._create_test_data()
        
        # Training statistics
        self.training_stats = {
            'round_times': [],
            'accuracies': [],
            'participation_rates': []
        }
    
    def _create_test_data(self) -> Dict[str, np.ndarray]:
        """Create test data for evaluation."""
        np.random.seed(42)
        num_samples = 1000
        input_dim = self.global_model.shape[1]
        num_classes = self.global_model.shape[0]
        
        return {
            'X': np.random.randn(num_samples, input_dim) * 0.5,
            'y': np.random.randint(0, num_classes, size=num_samples)
        }
    
    def run_training(self, num_rounds: int = 50, devices_per_round: int = 10, 
                     num_epochs: int = 1, selection_strategy: str = 'random',
                     async_mode: bool = False):
        """
        Run federated training.
        
        Args:
            num_rounds: Number of federated rounds
            devices_per_round: Number of devices per round
            num_epochs: Local epochs per round
            selection_strategy: Device selection strategy
            async_mode: Whether to use asynchronous training
        """
        print(f"Starting cloud-edge federated training with {len(self.devices)} devices")
        print(f"Rounds: {num_rounds}, Devices per round: {devices_per_round}")
        print(f"Selection strategy: {selection_strategy}, Async mode: {async_mode}")
        print("="*60)
        
        for round_idx in range(num_rounds):
            start_time = time.time()
            
            # Select devices
            selected = self.server.select_devices(devices_per_round, strategy=selection_strategy)
            
            # Execute round
            if async_mode:
                successful = self.server.async_round(selected, num_epochs=num_epochs)
            else:
                successful = self.server.round(selected, num_epochs=num_epochs)
            
            round_time = time.time() - start_time
            
            # Evaluate
            accuracy = self.server.evaluate(self.test_data)
            
            # Record statistics
            self.training_stats['round_times'].append(round_time)
            self.training_stats['accuracies'].append(accuracy)
            self.training_stats['participation_rates'].append(successful / len(selected) if selected else 0)
            
            print(f"Round {round_idx + 1}/{num_rounds}:")
            print(f"  Selected: {len(selected)}, Successful: {successful}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Time: {round_time:.2f}s")
        
        print("\nTraining complete!")
    
    def get_results(self) -> Dict[str, Any]:
        """Get training results."""
        return {
            'server_stats': self.server.get_stats(),
            'training_stats': self.training_stats,
            'round_history': self.server.get_round_history()
        }


# Example usage
if __name__ == "__main__":
    print("=== Testing Cloud Server ===")
    
    # Create coordinator
    coordinator = CloudEdgeCoordinator(num_devices=5, device_distribution='heterogeneous')
    
    # Run training
    coordinator.run_training(num_rounds=5, devices_per_round=3, num_epochs=1)
    
    # Get results
    results = coordinator.get_results()
    print(f"\nFinal accuracy: {results['training_stats']['accuracies'][-1]:.4f}")
    print(f"Total communication: {results['server_stats']['total_comm_cost']:.2f} MB")
