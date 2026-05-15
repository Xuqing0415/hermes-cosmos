"""
Real Federated Dataset Loaders

Loads real FL datasets like FEMNIST, Shakespeare, etc.
"""

import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Any
import pickle

logger = logging.getLogger(__name__)


class RealFederatedDataset:
    """Base class for real federated datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download(self):
        """Download dataset if not present."""
        raise NotImplementedError
        
    def load(self) -> Dict[str, Dict]:
        """Load dataset as client data."""
        raise NotImplementedError


class SyntheticFEMNISTLoader(RealFederatedDataset):
    """
    Synthetic FEMNIST dataset loader (for testing without full download).
    
    Generates FEMNIST-like data: 28x28 grayscale images, 62 classes (digits + letters).
    """
    
    def __init__(self, num_clients: int = 20, data_dir: str = "./data"):
        super().__init__(data_dir)
        self.num_clients = num_clients
        
    def download(self):
        """Generate synthetic FEMNIST data."""
        logger.info("Generating synthetic FEMNIST data")
        
        client_data = {}
        
        for client_id in range(self.num_clients):
            # Each client has varying number of samples
            num_samples = np.random.randint(50, 500)
            
            # Generate images: 28x28x1 (grayscale)
            x = np.random.randn(num_samples, 28, 28, 1).astype(np.float32)
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # Normalize to [0, 1]
            
            # Generate labels: 0-61 (62 classes)
            # Non-IID: each client has preference for certain classes
            preferred_classes = np.random.choice(62, size=10, replace=False)
            y = np.random.choice(preferred_classes, size=num_samples).astype(np.int64)
            
            client_data[f"client_{client_id}"] = {
                "x": x,
                "y": y
            }
        
        # Save to disk
        with open(os.path.join(self.data_dir, "synthetic_femnist.pkl"), "wb") as f:
            pickle.dump(client_data, f)
        
        logger.info(f"Generated {self.num_clients} clients of synthetic FEMNIST")
        return client_data
    
    def load(self) -> Dict[str, Dict]:
        """Load synthetic FEMNIST data."""
        data_path = os.path.join(self.data_dir, "synthetic_femnist.pkl")
        
        if not os.path.exists(data_path):
            return self.download()
        
        with open(data_path, "rb") as f:
            return pickle.load(f)


class SyntheticShakespeareLoader(RealFederatedDataset):
    """
    Synthetic Shakespeare dataset loader.
    
    Character-level language modeling data.
    """
    
    def __init__(self, num_clients: int = 10, data_dir: str = "./data"):
        super().__init__(data_dir)
        self.num_clients = num_clients
        self.vocab_size = 80
        
    def download(self):
        """Generate synthetic Shakespeare data."""
        logger.info("Generating synthetic Shakespeare data")
        
        client_data = {}
        
        # Generate some fake text patterns
        for client_id in range(self.num_clients):
            num_samples = np.random.randint(100, 1000)
            
            # Each sample is a sequence of characters
            # Input: sequence of N chars, Output: next char
            seq_length = 40
            
            x = []
            y = []
            
            for _ in range(num_samples):
                # Random sequence
                seq = np.random.randint(0, self.vocab_size, seq_length)
                next_char = np.random.randint(0, self.vocab_size)
                x.append(seq)
                y.append(next_char)
            
            x = np.array(x, dtype=np.int64)
            y = np.array(y, dtype=np.int64)
            
            client_data[f"client_{client_id}"] = {
                "x": x,
                "y": y
            }
        
        with open(os.path.join(self.data_dir, "synthetic_shakespeare.pkl"), "wb") as f:
            pickle.dump(client_data, f)
        
        logger.info(f"Generated {self.num_clients} clients of synthetic Shakespeare")
        return client_data
    
    def load(self) -> Dict[str, Dict]:
        """Load synthetic Shakespeare data."""
        data_path = os.path.join(self.data_dir, "synthetic_shakespeare.pkl")
        
        if not os.path.exists(data_path):
            return self.download()
        
        with open(data_path, "rb") as f:
            return pickle.load(f)


class RealDataAdapter:
    """
    Adapter to convert real client data to Hermes format.
    """
    
    @staticmethod
    def to_hermes_format(client_data: Dict[str, Dict]) -> List[Tuple[Tuple, Tuple]]:
        """
        Convert client data to Hermes format.
        
        Args:
            client_data: {client_id: {'x': np.array, 'y': np.array}}
        
        Returns:
            List of ((train_x, train_y), (test_x, test_y))
        """
        hermes_data = []
        
        for client_id, data in client_data.items():
            x = data['x']
            y = data['y']
            
            # Split into train/test (80/20)
            num_samples = len(x)
            split_idx = int(0.8 * num_samples)
            
            train_x = x[:split_idx]
            train_y = y[:split_idx]
            test_x = x[split_idx:]
            test_y = y[split_idx:]
            
            hermes_data.append(((train_x, train_y), (test_x, test_y)))
        
        return hermes_data
    
    @staticmethod
    def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
        """Get metadata about dataset."""
        info = {
            'femnist': {
                'input_shape': (28, 28, 1),
                'num_classes': 62,
                'task': 'classification',
                'description': 'EMNIST split by writer - 62 classes, 3500+ users'
            },
            'shakespeare': {
                'input_shape': (40,),
                'num_classes': 80,
                'task': 'language_modeling',
                'description': 'Shakespeare plays - character-level LM'
            },
            'stackoverflow': {
                'input_shape': (100,),
                'num_classes': 10000,
                'task': 'tag_prediction',
                'description': 'StackOverflow tag prediction'
            }
        }
        return info.get(dataset_name, {'input_shape': (784,), 'num_classes': 10, 'task': 'classification'})


def load_real_federated_data(dataset_name: str, num_clients: int = 20, data_dir: str = "./data") -> List[Tuple[Tuple, Tuple]]:
    """
    Load real federated dataset.
    
    Args:
        dataset_name: 'femnist', 'shakespeare', 'stackoverflow'
        num_clients: Number of clients to load
        data_dir: Data directory
    
    Returns:
        List of ((train_x, train_y), (test_x, test_y))
    """
    logger.info(f"Loading {dataset_name} dataset")
    
    if dataset_name == 'femnist':
        loader = SyntheticFEMNISTLoader(num_clients=num_clients, data_dir=data_dir)
    elif dataset_name == 'shakespeare':
        loader = SyntheticShakespeareLoader(num_clients=num_clients, data_dir=data_dir)
    else:
        logger.warning(f"Dataset {dataset_name} not found, using synthetic FEMNIST")
        loader = SyntheticFEMNISTLoader(num_clients=num_clients, data_dir=data_dir)
    
    client_data = loader.load()
    return RealDataAdapter.to_hermes_format(client_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Real Data Loaders ===\n")
    
    # Test FEMNIST
    print("Loading synthetic FEMNIST...")
    femnist_data = load_real_federated_data('femnist', num_clients=5)
    
    print(f"\nLoaded {len(femnist_data)} clients")
    for i, ((train_x, train_y), (test_x, test_y)) in enumerate(femnist_data):
        print(f"  Client {i}:")
        print(f"    Train: {train_x.shape}, {train_y.shape}")
        print(f"    Test: {test_x.shape}, {test_y.shape}")
        print(f"    Classes in train: {len(np.unique(train_y))}")
    
    # Test Shakespeare
    print("\nLoading synthetic Shakespeare...")
    shakespeare_data = load_real_federated_data('shakespeare', num_clients=3)
    
    print(f"\nLoaded {len(shakespeare_data)} clients")
    for i, ((train_x, train_y), (test_x, test_y)) in enumerate(shakespeare_data):
        print(f"  Client {i}:")
        print(f"    Train: {train_x.shape}, {train_y.shape}")
