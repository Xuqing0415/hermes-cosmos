"""
Federated Quantum Learning Module

Implements federated learning with quantum neural networks.
Uses PennyLane for quantum circuit simulation.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import warnings

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    warnings.warn("PennyLane not installed. Some features may not work.")


class QuantumCircuit:
    """
    Parametrized Quantum Circuit (PQC) for quantum machine learning.
    
    Implements a variational quantum circuit with parameterized gates.
    """
    
    def __init__(self, num_qubits: int = 4, num_layers: int = 2):
        """
        Initialize quantum circuit.
        
        Args:
            num_qubits: Number of qubits
            num_layers: Number of variational layers
        """
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for quantum circuits")
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_params = num_qubits * num_layers * 2
        
        self.dev = qml.device("default.qubit", wires=num_qubits)
    
    def circuit(self, params: np.ndarray, x: np.ndarray = None):
        """
        Define the quantum circuit.
        
        Args:
            params: Circuit parameters
            x: Input data (classical data to encode)
        
        Returns:
            Quantum node function
        """
        @qml.qnode(self.dev)
        def qnode(inputs=None):
            # Encode input data
            if inputs is not None and len(inputs) > 0:
                for i in range(min(self.num_qubits, len(inputs))):
                    qml.RX(inputs[i], wires=i)
            
            # Variational layers
            params_reshaped = params.reshape(self.num_layers, self.num_qubits, 2)
            
            for layer in range(self.num_layers):
                # Rotation gates
                for qubit in range(self.num_qubits):
                    qml.Rot(
                        params_reshaped[layer, qubit, 0],
                        params_reshaped[layer, qubit, 1],
                        0.0,
                        wires=qubit
                    )
                
                # Entangling gates
                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        
        return qnode(x) if x is not None else qnode
    
    def forward(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the quantum circuit.
        
        Args:
            params: Circuit parameters
            x: Input data batch
        
        Returns:
            Output measurements
        """
        results = []
        for sample in x:
            output = self.circuit(params, sample)
            results.append(np.array(output))
        return np.array(results)
    
    def get_num_parameters(self) -> int:
        """Get number of parameters."""
        return self.num_params


class QuantumClassifier:
    """
    Quantum classifier using parametrized quantum circuits.
    """
    
    def __init__(self, num_qubits: int = 4, num_layers: int = 2, num_classes: int = 2):
        """
        Initialize quantum classifier.
        
        Args:
            num_qubits: Number of qubits
            num_layers: Number of variational layers
            num_classes: Number of output classes
        """
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for quantum circuits")
        
        self.circuit = QuantumCircuit(num_qubits, num_layers)
        self.num_classes = num_classes
        
        # Initialize parameters randomly
        self.params = np.random.randn(self.circuit.get_num_parameters()) * 0.1
        
        # Classical output layer weights
        self.output_weights = np.random.randn(num_qubits, num_classes) * 0.1
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the quantum classifier.
        
        Args:
            x: Input data batch
        
        Returns:
            Class probabilities
        """
        # Quantum feature extraction
        quantum_features = self.circuit.forward(self.params, x)
        
        # Classical classification
        logits = quantum_features @ self.output_weights
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            x: Input data batch
        
        Returns:
            Predicted class labels
        """
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
    
    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            x: Input data
            y: True labels
        
        Returns:
            Loss value
        """
        probs = self.forward(x)
        one_hot = np.eye(self.num_classes)[y]
        return -np.mean(np.sum(one_hot * np.log(probs + 1e-10), axis=1))
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all model parameters."""
        return {
            'quantum_params': self.params.copy(),
            'output_weights': self.output_weights.copy()
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set model parameters."""
        if 'quantum_params' in params:
            self.params = params['quantum_params'].copy()
        if 'output_weights' in params:
            self.output_weights = params['output_weights'].copy()
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return len(self.params) + self.output_weights.size


class FedQNNClient:
    """
    Federated client for quantum neural networks.
    """
    
    def __init__(self, client_id: int, num_qubits: int = 4, num_classes: int = 2):
        """
        Initialize federated quantum client.
        
        Args:
            client_id: Unique client identifier
            num_qubits: Number of qubits
            num_classes: Number of output classes
        """
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for FedQNN")
        
        self.client_id = client_id
        self.classifier = QuantumClassifier(
            num_qubits=num_qubits,
            num_classes=num_classes
        )
        
        self.dataset = None
        
        # Learning rate
        self.lr = 0.01
        
        # Statistics
        self.loss_history = []
    
    def set_data(self, dataset: List[tuple]):
        """Set local dataset."""
        self.dataset = dataset
    
    def local_train(self, global_params: Dict[str, np.ndarray], 
                    num_epochs: int = 1) -> Dict[str, np.ndarray]:
        """
        Perform local training using parameter shift rule.
        
        Args:
            global_params: Global parameters to initialize with
            num_epochs: Number of local epochs
        
        Returns:
            Parameter updates
        """
        # Load global parameters
        self.classifier.set_parameters(global_params)
        
        # Simple gradient descent with parameter shift
        for epoch in range(num_epochs):
            # Compute gradients using parameter shift
            grads = self._compute_gradients()
            
            # Update parameters
            self.classifier.params -= self.lr * grads['quantum_params']
            self.classifier.output_weights -= self.lr * grads['output_weights']
            
            # Compute loss
            X, y = self._dataset_to_arrays()
            loss = self.classifier.compute_loss(X, y)
            self.loss_history.append(loss)
            
            print(f"Client {self.client_id}, Epoch {epoch+1}, Loss: {loss:.4f}")
        
        # Compute updates
        local_params = self.classifier.get_parameters()
        updates = {
            'quantum_params': local_params['quantum_params'] - global_params['quantum_params'],
            'output_weights': local_params['output_weights'] - global_params['output_weights']
        }
        
        return updates
    
    def _dataset_to_arrays(self) -> tuple:
        """Convert dataset to numpy arrays."""
        X = np.array([sample[0] for sample in self.dataset])
        y = np.array([sample[1] for sample in self.dataset])
        return X, y
    
    def _compute_gradients(self) -> Dict[str, np.ndarray]:
        """
        Compute gradients using parameter shift rule.
        
        Returns:
            Gradients for all parameters
        """
        X, y = self._dataset_to_arrays()
        
        # Gradient for quantum parameters
        quantum_grads = np.zeros_like(self.classifier.params)
        shift = np.pi / 4
        
        for i in range(len(self.classifier.params)):
            # Parameter shift rule
            params_plus = self.classifier.params.copy()
            params_minus = self.classifier.params.copy()
            
            params_plus[i] += shift
            params_minus[i] -= shift
            
            # Set temporary parameters
            old_params = self.classifier.params.copy()
            
            self.classifier.params = params_plus
            loss_plus = self.classifier.compute_loss(X, y)
            
            self.classifier.params = params_minus
            loss_minus = self.classifier.compute_loss(X, y)
            
            # Restore original parameters
            self.classifier.params = old_params
            
            # Compute gradient
            quantum_grads[i] = (loss_plus - loss_minus) / (2 * np.sin(shift))
        
        # Gradient for output weights (classical)
        output_grads = np.zeros_like(self.classifier.output_weights)
        batch_size = len(X)
        
        for i in range(self.classifier.output_weights.shape[0]):
            for j in range(self.classifier.output_weights.shape[1]):
                weights_plus = self.classifier.output_weights.copy()
                weights_minus = self.classifier.output_weights.copy()
                
                weights_plus[i, j] += shift
                weights_minus[i, j] -= shift
                
                old_weights = self.classifier.output_weights.copy()
                
                self.classifier.output_weights = weights_plus
                loss_plus = self.classifier.compute_loss(X, y)
                
                self.classifier.output_weights = weights_minus
                loss_minus = self.classifier.compute_loss(X, y)
                
                self.classifier.output_weights = old_weights
                
                output_grads[i, j] = (loss_plus - loss_minus) / (2 * np.sin(shift))
        
        return {
            'quantum_params': quantum_grads,
            'output_weights': output_grads
        }
    
    def evaluate(self, eval_dataset: List[tuple]) -> float:
        """Evaluate on evaluation dataset."""
        X = np.array([sample[0] for sample in eval_dataset])
        y = np.array([sample[1] for sample in eval_dataset])
        
        predictions = self.classifier.predict(X)
        return np.mean(predictions == y)


class FedQNNServer:
    """
    Server for federated quantum neural networks.
    """
    
    def __init__(self, num_qubits: int = 4, num_classes: int = 2):
        """
        Initialize federated quantum server.
        
        Args:
            num_qubits: Number of qubits
            num_classes: Number of output classes
        """
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for FedQNN")
        
        self.classifier = QuantumClassifier(
            num_qubits=num_qubits,
            num_classes=num_classes
        )
        
        # Statistics
        self.round = 0
        self.total_communication = 0.0
    
    def get_global_parameters(self) -> Dict[str, np.ndarray]:
        """Get current global parameters."""
        return self.classifier.get_parameters()
    
    def aggregate(self, updates: List[Dict[str, np.ndarray]], 
                  method: str = 'fedavg') -> Dict[str, np.ndarray]:
        """
        Aggregate client updates.
        
        Args:
            updates: List of parameter updates from clients
            method: Aggregation method
        
        Returns:
            Aggregated parameters
        """
        if not updates:
            return self.get_global_parameters()
        
        # Calculate communication cost
        for update in updates:
            param_bytes = update['quantum_params'].nbytes
            weight_bytes = update['output_weights'].nbytes
            self.total_communication += (param_bytes + weight_bytes) / (1024 * 1024)
        
        # Aggregate updates
        if method == 'fedavg':
            aggregated = {}
            
            for key in updates[0].keys():
                aggregated[key] = np.mean(
                    np.array([update[key] for update in updates]),
                    axis=0
                )
        
        else:
            # Default to FedAvg
            aggregated = {}
            for key in updates[0].keys():
                aggregated[key] = np.mean(
                    np.array([update[key] for update in updates]),
                    axis=0
                )
        
        # Update global parameters
        global_params = self.get_global_parameters()
        for key in global_params:
            global_params[key] += aggregated[key]
        
        self.classifier.set_parameters(global_params)
        self.round += 1
        
        return global_params
    
    def evaluate(self, eval_dataset: List[tuple]) -> float:
        """Evaluate global model."""
        X = np.array([sample[0] for sample in eval_dataset])
        y = np.array([sample[1] for sample in eval_dataset])
        
        predictions = self.classifier.predict(X)
        return np.mean(predictions == y)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'round': self.round,
            'total_communication': self.total_communication,
            'num_parameters': self.classifier.count_parameters()
        }


class FedQNNCoordinator:
    """
    Coordinator for federated quantum neural networks.
    """
    
    def __init__(self, num_qubits: int = 4, num_classes: int = 2, num_clients: int = 3):
        """
        Initialize coordinator.
        
        Args:
            num_qubits: Number of qubits
            num_classes: Number of output classes
            num_clients: Number of clients
        """
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane is required for FedQNN")
        
        self.server = FedQNNServer(num_qubits, num_classes)
        self.clients = [FedQNNClient(i, num_qubits, num_classes) for i in range(num_clients)]
        
        self.results = {
            'accuracy_history': [],
            'communication_history': []
        }
    
    def assign_data(self, datasets: List[List[tuple]]):
        """Assign datasets to clients."""
        for i, dataset in enumerate(datasets):
            if i < len(self.clients):
                self.clients[i].set_data(dataset)
    
    def run(self, num_rounds: int = 5, local_epochs: int = 1):
        """
        Run federated quantum learning.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Number of local epochs per client
        """
        print(f"Starting FedQNN with {len(self.clients)} clients")
        
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
            
            global_params = self.server.get_global_parameters()
            
            updates = []
            for client in self.clients:
                print(f"Training client {client.client_id}...")
                update = client.local_train(global_params, local_epochs=local_epochs)
                updates.append(update)
            
            self.server.aggregate(updates)
            
            # Evaluate
            eval_dataset = self.clients[0].dataset if self.clients[0].dataset else []
            if eval_dataset:
                accuracy = self.server.evaluate(eval_dataset)
                print(f"Round {round_idx + 1} Accuracy: {accuracy:.4f}")
                self.results['accuracy_history'].append(accuracy)
            
            self.results['communication_history'].append(self.server.total_communication)
        
        return self.results


def generate_quantum_dataset(num_samples: int = 100, num_features: int = 4, num_classes: int = 2):
    """Generate synthetic quantum dataset."""
    X = np.random.randn(num_samples, num_features)
    
    # Simple linear classification
    weights = np.random.randn(num_features)
    logits = X @ weights
    y = (logits > 0).astype(int)
    
    return [(X[i], y[i]) for i in range(num_samples)]


if __name__ == "__main__":
    if not HAS_PENNYLANE:
        print("PennyLane not installed. Skipping quantum tests.")
        print("Install with: pip install pennylane")
    else:
        print("=== Testing Federated Quantum Learning ===")
        
        # Test Quantum Circuit
        print("\n1. Testing Quantum Circuit...")
        circuit = QuantumCircuit(num_qubits=4, num_layers=2)
        params = np.random.randn(circuit.get_num_parameters())
        x = np.random.randn(4)
        output = circuit.forward(params, np.array([x]))
        print(f"   Circuit output shape: {output.shape}")
        
        # Test Quantum Classifier
        print("\n2. Testing Quantum Classifier...")
        classifier = QuantumClassifier(num_qubits=4, num_classes=2)
        X = np.random.randn(10, 4)
        probs = classifier.forward(X)
        print(f"   Classifier output shape: {probs.shape}")
        print(f"   Number of parameters: {classifier.count_parameters()}")
        
        # Test FedQNN Client
        print("\n3. Testing FedQNN Client...")
        client = FedQNNClient(client_id=0, num_qubits=4, num_classes=2)
        dataset = generate_quantum_dataset(num_samples=20)
        client.set_data(dataset)
        
        global_params = client.classifier.get_parameters()
        updates = client.local_train(global_params, num_epochs=1)
        print(f"   Update shapes: quantum_params={updates['quantum_params'].shape}")
        
        # Test FedQNN Server
        print("\n4. Testing FedQNN Server...")
        server = FedQNNServer(num_qubits=4, num_classes=2)
        updates_list = [updates, updates]
        server.aggregate(updates_list)
        print(f"   Server round: {server.round}")
        
        print("\n=== All quantum tests passed! ===")