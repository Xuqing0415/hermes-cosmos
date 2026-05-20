"""
Incentive Mechanisms for Federated Learning

Implements token distribution and reputation systems based on Shapley values.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time


class TokenDistributor:
    """Distributes tokens based on Shapley values."""
    
    def __init__(self, total_tokens_per_round: int = 1000):
        self.total_tokens = total_tokens_per_round
        self.token_balances: Dict[int, float] = defaultdict(float)
        self.token_history: Dict[int, List[float]] = defaultdict(list)
    
    def distribute(self, shapley_values: Dict[int, float]):
        """
        Distribute tokens based on Shapley values.
        
        Args:
            shapley_values: Dictionary of client Shapley values
        """
        total_value = sum(shapley_values.values())
        
        if total_value == 0:
            equal_share = self.total_tokens / len(shapley_values)
            for client_id in shapley_values:
                self.token_balances[client_id] += equal_share
                self.token_history[client_id].append(equal_share)
            return
        
        for client_id, value in shapley_values.items():
            tokens = (value / total_value) * self.total_tokens
            self.token_balances[client_id] += tokens
            self.token_history[client_id].append(tokens)
    
    def get_balance(self, client_id: int) -> float:
        """Get token balance for a client."""
        return self.token_balances[client_id]
    
    def spend_tokens(self, client_id: int, amount: float) -> bool:
        """
        Spend tokens for a client.
        
        Args:
            client_id: Client ID
            amount: Amount to spend
        
        Returns:
            True if successful
        """
        if self.token_balances[client_id] >= amount:
            self.token_balances[client_id] -= amount
            return True
        return False
    
    def get_total_distributed(self) -> float:
        """Get total tokens distributed."""
        return sum(self.token_balances.values())


class ReputationSystem:
    """Maintains reputation scores for clients."""
    
    def __init__(self, decay_factor: float = 0.99):
        self.decay_factor = decay_factor
        self.reputations: Dict[int, float] = defaultdict(float)
        self.history: Dict[int, List[float]] = defaultdict(list)
    
    def update_reputation(self, client_id: int, contribution: float):
        """
        Update reputation based on contribution.
        
        Args:
            client_id: Client ID
            contribution: Contribution score
        """
        self.reputations[client_id] = (
            self.decay_factor * self.reputations[client_id] +
            (1 - self.decay_factor) * contribution
        )
        
        self.history[client_id].append(self.reputations[client_id])
    
    def get_reputation(self, client_id: int) -> float:
        """Get reputation for a client."""
        return self.reputations[client_id]
    
    def get_top_clients(self, n: int = 10) -> List[Tuple[int, float]]:
        """Get top N clients by reputation."""
        sorted_clients = sorted(
            self.reputations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_clients[:n]
    
    def reset_reputation(self, client_id: int):
        """Reset reputation for a client."""
        self.reputations[client_id] = 0.0


class IncentiveMechanism:
    """Combined incentive mechanism using Shapley values."""
    
    def __init__(self, token_distributor: TokenDistributor,
                 reputation_system: ReputationSystem):
        self.token_distributor = token_distributor
        self.reputation_system = reputation_system
        
        self.participation_records: Dict[int, List[float]] = defaultdict(list)
    
    def reward_contribution(self, client_id: int, shapley_value: float):
        """
        Reward a client based on their Shapley value contribution.
        
        Args:
            client_id: Client ID
            shapley_value: Shapley value
        """
        self.token_distributor.distribute({client_id: shapley_value})
        self.reputation_system.update_reputation(client_id, shapley_value)
        
        self.participation_records[client_id].append({
            'timestamp': time.time(),
            'shapley_value': shapley_value,
            'tokens_earned': shapley_value * 1000,
            'reputation': self.reputation_system.get_reputation(client_id)
        })
    
    def reward_all(self, shapley_values: Dict[int, float]):
        """Reward all clients based on their Shapley values."""
        self.token_distributor.distribute(shapley_values)
        
        for client_id, value in shapley_values.items():
            self.reputation_system.update_reputation(client_id, value)
            self.participation_records[client_id].append({
                'timestamp': time.time(),
                'shapley_value': value,
                'reputation': self.reputation_system.get_reputation(client_id)
            })
    
    def get_client_stats(self, client_id: int) -> Dict[str, Any]:
        """Get statistics for a client."""
        return {
            'client_id': client_id,
            'tokens': self.token_distributor.get_balance(client_id),
            'reputation': self.reputation_system.get_reputation(client_id),
            'participations': len(self.participation_records[client_id])
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global incentive statistics."""
        return {
            'total_tokens_distributed': self.token_distributor.get_total_distributed(),
            'num_clients_participated': len(self.participation_records),
            'avg_reputation': np.mean(list(self.reputation_system.reputations.values())) if self.reputation_system.reputations else 0.0
        }


class ContributionTracker:
    """Tracks client contributions over time."""
    
    def __init__(self):
        self.contributions: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    
    def record_contribution(self, client_id: int, round_idx: int,
                          shapley_value: float, data_quality: float = 1.0):
        """
        Record a client's contribution.
        
        Args:
            client_id: Client ID
            round_idx: Round index
            shapley_value: Shapley value
            data_quality: Data quality score
        """
        self.contributions[client_id].append({
            'round': round_idx,
            'shapley_value': shapley_value,
            'data_quality': data_quality,
            'timestamp': time.time()
        })
    
    def get_contribution_history(self, client_id: int) -> List[Dict[str, float]]:
        """Get contribution history for a client."""
        return self.contributions[client_id]
    
    def get_cumulative_shapley(self, client_id: int) -> float:
        """Get cumulative Shapley value for a client."""
        return sum(c['shapley_value'] for c in self.contributions[client_id])
    
    def detect_anomalous_contribution(self, client_id: int,
                                     threshold: float = -0.1) -> bool:
        """
        Detect anomalous contributions.
        
        Args:
            client_id: Client ID
            threshold: Threshold for anomaly detection
        
        Returns:
            True if anomalous
        """
        recent = self.contributions[client_id][-10:]
        
        if len(recent) < 3:
            return False
        
        recent_values = [c['shapley_value'] for c in recent]
        mean_value = np.mean(recent_values)
        
        return mean_value < threshold


class SmartContractSimulator:
    """Simulates a smart contract for token distribution."""
    
    def __init__(self, incentive_mechanism: IncentiveMechanism):
        self.incentive_mechanism = incentive_mechanism
        self.block_height = 0
    
    def mint_tokens(self, amount: float):
        """Mint new tokens."""
        self.incentive_mechanism.token_distributor.total_tokens += amount
    
    def execute_round(self, shapley_values: Dict[int, float]):
        """Execute a round of token distribution."""
        self.incentive_mechanism.reward_all(shapley_values)
        self.block_height += 1
    
    def get_block_height(self) -> int:
        """Get current block height."""
        return self.block_height
    
    def get_contract_balance(self) -> float:
        """Get total tokens in circulation."""
        return self.incentive_mechanism.token_distributor.get_total_distributed()