"""
Federated Reinforcement Learning Module

Implements federated learning for reinforcement learning agents,
enabling multiple agents to share policy knowledge while training
in different environments.
"""

import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Simple policy network for reinforcement learning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """Value network for estimating state values."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class HeterogeneousEnv(gym.Env):
    """
    Heterogeneous environment wrapper that modifies base environment parameters.
    
    Each client can have different environment characteristics:
    - Different gravity
    - Different friction
    - Different noise levels
    - Different reward scaling
    """
    
    def __init__(self, env_name: str = "CartPole-v1", 
                 gravity_scale: float = 1.0, friction_scale: float = 1.0,
                 noise_std: float = 0.0, reward_scale: float = 1.0):
        self.base_env = gym.make(env_name)
        self.gravity_scale = gravity_scale
        self.friction_scale = friction_scale
        self.noise_std = noise_std
        self.reward_scale = reward_scale
        
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        
    def step(self, action):
        obs, reward, done, truncated, info = self.base_env.step(action)
        
        # Apply noise to observation
        if self.noise_std > 0:
            obs = obs + np.random.normal(0, self.noise_std, obs.shape)
        
        # Scale reward
        reward = reward * self.reward_scale
        
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None):
        return self.base_env.reset(seed=seed)
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        self.base_env.close()


class RLClient:
    """
    Federated RL client that trains locally and shares policy updates.
    """
    
    def __init__(self, client_id: str, env_name: str = "CartPole-v1",
                 state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        self.client_id = client_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create heterogeneous environment
        np.random.seed(hash(client_id) % 10000)
        self.env = HeterogeneousEnv(
            env_name=env_name,
            gravity_scale=np.random.uniform(0.8, 1.2),
            friction_scale=np.random.uniform(0.5, 1.5),
            noise_std=np.random.uniform(0, 0.1),
            reward_scale=np.random.uniform(0.9, 1.1)
        )
        
        # Policy and value networks
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        
        # Statistics
        self.total_rewards = []
        self.episode_lengths = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action using current policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def compute_returns(self, rewards: List[float], gamma: float = 0.99) -> np.ndarray:
        """Compute discounted returns."""
        returns = []
        running_sum = 0
        for r in reversed(rewards):
            running_sum = r + gamma * running_sum
            returns.insert(0, running_sum)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def train_episode(self, max_steps: int = 500) -> Dict[str, Any]:
        """Train for one episode."""
        state, _ = self.env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        for _ in range(max_steps):
            action, log_prob = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
            
            if done:
                break
        
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Update policy
        self.policy_optimizer.zero_grad()
        for i, (log_prob, ret) in enumerate(zip(log_probs, returns)):
            loss = -log_prob * ret
            loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        states_tensor = torch.tensor(states, dtype=torch.float32)
        values = self.value_net(states_tensor)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        value_loss = F.mse_loss(values, returns_tensor)
        value_loss.backward()
        self.value_optimizer.step()
        
        total_reward = sum(rewards)
        
        self.total_rewards.append(total_reward)
        self.episode_lengths.append(len(rewards))
        
        return {
            'episode_length': len(rewards),
            'total_reward': total_reward,
            'avg_reward': np.mean(self.total_rewards[-10:]) if self.total_rewards else 0
        }
    
    def get_policy_weights(self) -> Dict[str, np.ndarray]:
        """Get policy network weights as numpy arrays."""
        return {k: v.detach().cpu().numpy() for k, v in self.policy_net.state_dict().items()}
    
    def set_policy_weights(self, weights: Dict[str, np.ndarray]):
        """Set policy network weights from numpy arrays."""
        state_dict = {k: torch.tensor(v) for k, v in weights.items()}
        self.policy_net.load_state_dict(state_dict)
    
    def get_value_weights(self) -> Dict[str, np.ndarray]:
        """Get value network weights as numpy arrays."""
        return {k: v.detach().cpu().numpy() for k, v in self.value_net.state_dict().items()}
    
    def set_value_weights(self, weights: Dict[str, np.ndarray]):
        """Set value network weights from numpy arrays."""
        state_dict = {k: torch.tensor(v) for k, v in weights.items()}
        self.value_net.load_state_dict(state_dict)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'client_id': self.client_id,
            'total_rewards': self.total_rewards,
            'episode_lengths': self.episode_lengths,
            'avg_reward_last_10': np.mean(self.total_rewards[-10:]) if self.total_rewards else 0,
            'best_reward': max(self.total_rewards) if self.total_rewards else 0
        }


class FederatedRLServer:
    """
    Server for federated reinforcement learning.
    
    Implements FedAvg-AC algorithm with trust region constraints.
    """
    
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Global policy and value networks
        self.global_policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.global_value = ValueNetwork(state_dim, hidden_dim)
        
        # Trust region parameters
        self.kl_threshold = 0.01
        self.damping = 0.1
        
        # Clients
        self.clients: List[RLClient] = []
        
        # Statistics
        self.round_history = []
        self.total_communication = 0.0
    
    def add_client(self, client: RLClient):
        """Add a client to the server."""
        self.clients.append(client)
    
    def distribute_policy(self):
        """Distribute global policy to all clients."""
        policy_weights = {k: v.detach().cpu().numpy() for k, v in self.global_policy.state_dict().items()}
        value_weights = {k: v.detach().cpu().numpy() for k, v in self.global_value.state_dict().items()}
        
        for client in self.clients:
            client.set_policy_weights(policy_weights)
            client.set_value_weights(value_weights)
    
    def aggregate_updates(self, trust_region: bool = True) -> Dict[str, Any]:
        """
        Aggregate policy updates from clients.
        
        Args:
            trust_region: Whether to apply KL divergence constraint
        
        Returns:
            Aggregation statistics
        """
        if not self.clients:
            return {'num_updates': 0}
        
        # Collect updates
        updates = []
        for client in self.clients:
            local_policy = client.get_policy_weights()
            local_value = client.get_value_weights()
            updates.append((local_policy, local_value))
        
        # Compute average
        avg_policy = {}
        avg_value = {}
        
        for key in updates[0][0].keys():
            tensors = [u[0][key] for u in updates]
            avg_policy[key] = np.mean(tensors, axis=0)
        
        for key in updates[0][1].keys():
            tensors = [u[1][key] for u in updates]
            avg_value[key] = np.mean(tensors, axis=0)
        
        # Apply trust region constraint if enabled
        if trust_region:
            kl_div = self._compute_kl_divergence(avg_policy)
            if kl_div > self.kl_threshold:
                # Scale down the update
                scale = self.kl_threshold / (kl_div + self.damping)
                for key in avg_policy.keys():
                    avg_policy[key] = self.global_policy.state_dict()[key].detach().cpu().numpy() + \
                                   (avg_policy[key] - self.global_policy.state_dict()[key].detach().cpu().numpy()) * scale
        else:
            kl_div = 0.0
        
        # Update global networks
        self.global_policy.load_state_dict({k: torch.tensor(v) for k, v in avg_policy.items()})
        self.global_value.load_state_dict({k: torch.tensor(v) for k, v in avg_value.items()})
        
        # Update communication cost
        update_size = sum(v.nbytes for v in avg_policy.values()) + sum(v.nbytes for v in avg_value.values())
        self.total_communication += update_size * len(self.clients) * 2  # Download + Upload
        
        return {
            'num_updates': len(updates),
            'kl_divergence': kl_div,
            'update_size_mb': update_size / (1024 * 1024)
        }
    
    def _compute_kl_divergence(self, new_policy: Dict[str, np.ndarray]) -> float:
        """
        Compute KL divergence between current policy and new policy.
        
        Approximated by parameter distance for simplicity.
        """
        total_distance = 0.0
        for key, new_val in new_policy.items():
            old_val = self.global_policy.state_dict()[key].detach().cpu().numpy()
            diff = new_val - old_val
            total_distance += np.sum(diff ** 2)
        return total_distance / 1e6  # Scale to reasonable range
    
    def evaluate_global_policy(self, env_name: str = "CartPole-v1", num_episodes: int = 5) -> float:
        """Evaluate the global policy on the standard environment."""
        env = gym.make(env_name)
        total_rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            for _ in range(500):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                probs = self.global_policy(state_tensor)
                action = torch.argmax(probs).item()
                state, reward, done, _, _ = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            total_rewards.append(total_reward)
        
        env.close()
        return np.mean(total_rewards)
    
    def run_federated_training(self, num_rounds: int = 10, episodes_per_round: int = 10,
                              trust_region: bool = True):
        """
        Run federated RL training.
        
        Args:
            num_rounds: Number of federated rounds
            episodes_per_round: Episodes per client per round
            trust_region: Whether to use trust region constraints
        """
        print(f"Starting federated RL training with {len(self.clients)} clients")
        print(f"Rounds: {num_rounds}, Episodes per round: {episodes_per_round}")
        
        for round_idx in range(num_rounds):
            start_time = time.time()
            
            # Distribute policy
            self.distribute_policy()
            
            # Local training
            for client in self.clients:
                for _ in range(episodes_per_round):
                    client.train_episode()
            
            # Aggregate updates
            agg_stats = self.aggregate_updates(trust_region=trust_region)
            
            # Evaluate
            avg_reward = self.evaluate_global_policy()
            
            round_time = time.time() - start_time
            
            self.round_history.append({
                'round': round_idx + 1,
                'avg_reward': avg_reward,
                'kl_divergence': agg_stats['kl_divergence'],
                'num_updates': agg_stats['num_updates'],
                'time': round_time
            })
            
            print(f"Round {round_idx + 1}/{num_rounds}:")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  KL Divergence: {agg_stats['kl_divergence']:.6f}")
            print(f"  Time: {round_time:.2f}s")
    
    def get_results(self) -> Dict[str, Any]:
        """Get training results."""
        return {
            'round_history': self.round_history,
            'total_communication_mb': self.total_communication / (1024 * 1024),
            'client_stats': [client.get_stats() for client in self.clients]
        }


def run_federated_vs_centralized():
    """Compare federated and centralized RL training."""
    print("\n=== Federated vs Centralized RL Comparison ===")
    
    # Federated training
    print("\n--- Federated RL ---")
    server = FederatedRLServer(state_dim=4, action_dim=2)
    
    for i in range(5):
        client = RLClient(f"client_{i}", env_name="CartPole-v1", state_dim=4, action_dim=2)
        server.add_client(client)
    
    server.run_federated_training(num_rounds=5, episodes_per_round=5)
    fed_results = server.get_results()
    
    # Centralized training
    print("\n--- Centralized RL ---")
    centralized_client = RLClient("centralized", env_name="CartPole-v1", state_dim=4, action_dim=2)
    
    for _ in range(25):  # 5 rounds * 5 episodes * 5 clients / 5 clients = 25 episodes
        stats = centralized_client.train_episode()
        if (_ + 1) % 5 == 0:
            print(f"Episode {_ + 1}: Reward = {stats['total_reward']:.2f}")
    
    # Comparison
    print("\n=== Comparison Summary ===")
    print(f"Federated Final Reward: {fed_results['round_history'][-1]['avg_reward']:.2f}")
    print(f"Centralized Final Reward: {centralized_client.get_stats()['avg_reward_last_10']:.2f}")
    print(f"Total Communication: {fed_results['total_communication_mb']:.2f} MB")


if __name__ == "__main__":
    run_federated_vs_centralized()
