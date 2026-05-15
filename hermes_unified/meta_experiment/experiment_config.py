"""
Experiment Configuration Module

Defines parameter space for automated experimentation.
"""

import yaml
import itertools
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Represents a single experiment configuration."""
    attack_type: str
    attack_intensity: float
    malicious_ratio: float
    defense_type: str
    num_clients: int
    non_iid_alpha: float
    num_rounds: int = 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'attack_type': self.attack_type,
            'attack_intensity': self.attack_intensity,
            'malicious_ratio': self.malicious_ratio,
            'defense_type': self.defense_type,
            'num_clients': self.num_clients,
            'non_iid_alpha': self.non_iid_alpha,
            'num_rounds': self.num_rounds
        }
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration."""
        return f"{self.attack_type}_{self.attack_intensity}_{self.malicious_ratio}_{self.defense_type}_{self.num_clients}_{self.non_iid_alpha}"


class ParameterSpace:
    """Defines the parameter space for experiments."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize parameter space.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.parameters = self._load_default_parameters()
        
        if config_file:
            self._load_from_file(config_file)
    
    def _load_default_parameters(self) -> Dict[str, List[Any]]:
        """Load default parameter ranges."""
        return {
            'attack_type': ['gradient_scale', 'label_flip', 'sign_flipping', 'backdoor'],
            'attack_intensity': [1.0, 2.0, 5.0, 10.0],
            'malicious_ratio': [0.0, 0.1, 0.2, 0.3, 0.4],
            'defense_type': ['krum', 'trimmed_mean', 'median', 'fedavg'],
            'num_clients': [10, 20, 50],
            'non_iid_alpha': [0.0, 0.3, 0.7, 1.0],  # 0 = IID, 1 = highly non-IID
            'num_rounds': [50, 100]
        }
    
    def _load_from_file(self, config_file: str):
        """Load parameters from YAML file."""
        with open(config_file, 'r') as f:
            loaded = yaml.safe_load(f)
            self.parameters.update(loaded)
    
    def generate_all_combinations(self) -> List[ExperimentConfig]:
        """Generate all possible parameter combinations."""
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        
        combinations = []
        for combo in itertools.product(*values):
            config_dict = dict(zip(keys, combo))
            config = ExperimentConfig(**config_dict)
            combinations.append(config)
        
        return combinations
    
    def sample_random(self, num_samples: int = 10) -> List[ExperimentConfig]:
        """Sample random configurations without replacement."""
        all_combinations = self.generate_all_combinations()
        return random.sample(all_combinations, min(num_samples, len(all_combinations)))
    
    def sample_unexplored(self, explored_hashes: List[str], num_samples: int = 10) -> List[ExperimentConfig]:
        """Sample configurations that haven't been explored yet."""
        all_combinations = self.generate_all_combinations()
        unexplored = [c for c in all_combinations if c.get_hash() not in explored_hashes]
        
        return random.sample(unexplored, min(num_samples, len(unexplored)))
    
    def get_total_combinations(self) -> int:
        """Calculate total number of possible combinations."""
        total = 1
        for values in self.parameters.values():
            total *= len(values)
        return total


def load_config_from_yaml(file_path: str) -> ParameterSpace:
    """Load parameter space from YAML file."""
    return ParameterSpace(file_path)


def save_config_to_yaml(parameter_space: ParameterSpace, file_path: str):
    """Save parameter space to YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(parameter_space.parameters, f, default_flow_style=False)


# Example usage
if __name__ == "__main__":
    ps = ParameterSpace()
    print(f"Total combinations: {ps.get_total_combinations()}")
    
    sample = ps.sample_random(5)
    print(f"\nRandom sample:")
    for i, config in enumerate(sample):
        print(f"{i+1}. {config.get_hash()}")
    
    # Test unexplored sampling
    explored = [c.get_hash() for c in sample]
    unexplored_sample = ps.sample_unexplored(explored, 3)
    print(f"\nUnexplored sample:")
    for i, config in enumerate(unexplored_sample):
        print(f"{i+1}. {config.get_hash()}")
