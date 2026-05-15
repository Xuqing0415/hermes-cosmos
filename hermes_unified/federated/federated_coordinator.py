"""
Federated Learning Game Coordinator

Implements game-theoretic coordination between attackers and defenders.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import time


class AdaptiveDefenseSelector:
    """
    Adaptive defense selector that chooses the best defense strategy based on real-time statistics.
    
    Maintains a list of defense candidates and selects based on:
    - Gradient norm variance
    - Anomaly ratio (proportion of updates that are outliers)
    - Communication delay
    - Accuracy change
    """
    
    def __init__(self, defense_candidates: List[str] = None):
        """
        Initialize adaptive defense selector.
        
        Args:
            defense_candidates: List of defense algorithms to consider
        """
        self.candidates = defense_candidates or ['krum', 'multi_krum', 'trimmed_mean', 'median']
        self.performance = {d: [] for d in self.candidates}
        self.current_defense = 'trimmed_mean'
        self.stats_history = []
        
        # Thresholds for decision making
        self.grad_var_high = 0.5
        self.anomaly_ratio_high = 0.3
        self.accuracy_drop_threshold = 0.05
    
    def select_defense(self, recent_stats: Dict) -> str:
        """
        Select the best defense based on recent statistics.
        
        Args:
            recent_stats: Dictionary containing statistics like grad_var, anomaly_ratio, accuracy_change
        
        Returns:
            Selected defense algorithm name
        """
        self.stats_history.append(recent_stats)
        
        grad_var = recent_stats.get('grad_var', 0.0)
        anomaly_ratio = recent_stats.get('anomaly_ratio', 0.0)
        accuracy_change = recent_stats.get('accuracy_change', 0.0)
        
        # Decision logic
        if grad_var > self.grad_var_high or anomaly_ratio > self.anomaly_ratio_high:
            # High variance or many anomalies: use robust defense
            if anomaly_ratio > 0.3:
                selected = 'multi_krum'
            else:
                selected = 'krum'
        elif accuracy_change < -self.accuracy_drop_threshold:
            # Accuracy dropped significantly: switch to more robust defense
            current_idx = self.candidates.index(self.current_defense)
            if current_idx < len(self.candidates) - 1:
                selected = self.candidates[current_idx + 1]
            else:
                selected = self.current_defense
        else:
            # Normal conditions: use efficient defense
            selected = 'median' if anomaly_ratio < 0.1 else 'trimmed_mean'
        
        self.current_defense = selected
        return selected
    
    def record_performance(self, defense: str, accuracy_change: float):
        """Record performance of a defense."""
        if defense in self.performance:
            self.performance[defense].append(accuracy_change)
    
    def get_best_defense(self) -> str:
        """Get defense with best historical performance."""
        best_defense = self.current_defense
        best_performance = float('-inf')
        
        for defense, changes in self.performance.items():
            if changes:
                avg_change = np.mean(changes)
                if avg_change > best_performance:
                    best_performance = avg_change
                    best_defense = defense
        
        return best_defense
    
    def get_stats(self) -> Dict:
        """Get selector statistics."""
        return {
            'current_defense': self.current_defense,
            'candidates': self.candidates,
            'stats_history_length': len(self.stats_history),
            'performance': {k: len(v) for k, v in self.performance.items()}
        }


class AdaptiveAttackClient:
    """
    Attack client that adapts its attack parameters using hill-climbing.
    
    Adjusts attack intensity based on observed impact on global accuracy.
    """
    
    def __init__(self, client_id: int, attack_type: str = 'gradient_scale', 
                 initial_scale: float = 5.0):
        """
        Initialize adaptive attack client.
        
        Args:
            client_id: Unique client identifier
            attack_type: Type of attack ('gradient_scale', 'label_flip', 'backdoor')
            initial_scale: Initial attack intensity
        """
        self.client_id = client_id
        self.attack_type = attack_type
        self.scale = initial_scale
        self.best_scale = initial_scale
        self.best_impact = 0.0
        self.exploration_prob = 0.2
        self.history = []
    
    def adapt_attack(self, global_accuracy_drop: float):
        """
        Adapt attack parameters using hill-climbing strategy.
        
        Args:
            global_accuracy_drop: Observed accuracy drop from previous round
        """
        self.history.append({
            'scale': self.scale,
            'impact': global_accuracy_drop
        })
        
        # Hill climbing with exploration
        if random.random() < self.exploration_prob:
            # Explore: random adjustment
            self.scale *= random.uniform(0.8, 1.3)
        else:
            # Exploit: use hill climbing
            if global_accuracy_drop > self.best_impact:
                self.best_impact = global_accuracy_drop
                self.best_scale = self.scale
                # Increase attack intensity
                self.scale *= 1.2
            else:
                # Decrease intensity, but keep some memory of best
                self.scale = self.best_scale * 0.95
        
        # Keep scale within bounds
        self.scale = max(1.0, min(20.0, self.scale))
    
    def get_attack_params(self) -> Dict:
        """Get current attack parameters."""
        return {
            'client_id': self.client_id,
            'attack_type': self.attack_type,
            'scale': self.scale,
            'best_scale': self.best_scale,
            'best_impact': self.best_impact
        }


class GameCoordinator:
    """
    Game coordinator that orchestrates the attack-defense game.
    
    Responsibilities:
    - Coordinate defense selection
    - Coordinate attack adaptation
    - Record game history
    - Visualize game progress
    """
    
    def __init__(self, num_clients: int = 20, num_malicious: int = 3, 
                 model_shape: Tuple[int, int] = (10, 784)):
        """
        Initialize game coordinator.
        
        Args:
            num_clients: Total number of clients
            num_malicious: Number of malicious clients
            model_shape: Shape of model parameters
        """
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.model_shape = model_shape
        
        # Game state
        self.global_model = np.random.randn(*model_shape) * 0.01
        self.round = 0
        self.game_over = False
        
        # Players
        self.defense_selector = AdaptiveDefenseSelector()
        self.malicious_clients = [
            AdaptiveAttackClient(client_id=i, attack_type='gradient_scale')
            for i in range(num_malicious)
        ]
        
        # Statistics
        self.accuracy_history = []
        self.defense_history = []
        self.attack_params_history = []
        self.game_log = []
        
        # Scores
        self.attack_score = 0
        self.defense_score = 0
    
    def _generate_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data."""
        X = np.random.randn(num_samples, self.model_shape[1])
        y = np.argmax(X @ np.random.randn(*self.model_shape).T, axis=1)
        return X, y
    
    def _evaluate(self) -> float:
        """Evaluate current model accuracy."""
        X_test, y_test = self._generate_data(num_samples=500)
        pred = X_test @ self.global_model.T
        predictions = np.argmax(pred, axis=1)
        return np.mean(predictions == y_test)
    
    def _compute_stats(self, updates: List[np.ndarray]) -> Dict:
        """Compute statistics from client updates."""
        if len(updates) < 2:
            return {
                'grad_var': 0.0,
                'anomaly_ratio': 0.0,
                'avg_norm': 0.0
            }
        
        norms = np.array([np.linalg.norm(u) for u in updates])
        grad_var = np.var(norms)
        avg_norm = np.mean(norms)
        
        # Detect anomalies based on z-score
        std_norm = np.std(norms) if len(norms) > 1 else 1.0
        z_scores = np.abs((norms - avg_norm) / (std_norm + 1e-10))
        anomaly_ratio = np.mean(z_scores > 2.0)
        
        return {
            'grad_var': grad_var,
            'anomaly_ratio': anomaly_ratio,
            'avg_norm': avg_norm,
            'num_updates': len(updates)
        }
    
    def _generate_updates(self) -> List[np.ndarray]:
        """Generate updates from all clients."""
        updates = []
        
        # Honest clients
        for _ in range(self.num_clients - self.num_malicious):
            update = np.random.randn(*self.model_shape) * 0.01
            updates.append(update)
        
        # Malicious clients
        for mc in self.malicious_clients:
            scale = mc.scale
            if mc.attack_type == 'gradient_scale':
                update = np.random.randn(*self.model_shape) * scale * 0.1
            else:
                update = np.random.randn(*self.model_shape) * 0.5
            updates.append(update)
        
        return updates
    
    def play_round(self) -> Dict:
        """Play one round of the game."""
        self.round += 1
        
        # Generate updates
        updates = self._generate_updates()
        
        # Compute statistics
        stats = self._compute_stats(updates)
        
        # Select defense
        prev_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0.0
        accuracy_change = 0.0
        
        if self.round > 1:
            accuracy_change = self.accuracy_history[-1] - prev_accuracy
        
        stats['accuracy_change'] = accuracy_change
        selected_defense = self.defense_selector.select_defense(stats)
        
        # Aggregate using selected defense
        aggregated_update = self._aggregate(updates, selected_defense)
        
        # Update global model
        self.global_model += aggregated_update * 0.01
        
        # Evaluate
        accuracy = self._evaluate()
        self.accuracy_history.append(accuracy)
        self.defense_history.append(selected_defense)
        
        # Record attack parameters
        attack_params = [mc.get_attack_params() for mc in self.malicious_clients]
        self.attack_params_history.append(attack_params)
        
        # Calculate accuracy drop from previous round
        acc_drop = 0.0
        if self.round > 1:
            acc_drop = max(0.0, self.accuracy_history[-2] - accuracy)
        
        # Attackers adapt
        for mc in self.malicious_clients:
            mc.adapt_attack(acc_drop)
        
        # Determine round winner
        if self.round > 1:
            if acc_drop > 0.05:
                self.attack_score += 1
                result = 'ATTACK WIN'
            elif accuracy > self.accuracy_history[-2] + 0.02:
                self.defense_score += 1
                result = 'DEFENSE WIN'
            else:
                result = 'DRAW'
        else:
            result = 'INITIAL'
        
        # Log round
        self.game_log.append({
            'round': self.round,
            'defense': selected_defense,
            'accuracy': accuracy,
            'accuracy_drop': acc_drop,
            'grad_var': stats['grad_var'],
            'anomaly_ratio': stats['anomaly_ratio'],
            'result': result
        })
        
        return {
            'round': self.round,
            'defense': selected_defense,
            'accuracy': accuracy,
            'accuracy_drop': acc_drop,
            'attack_score': self.attack_score,
            'defense_score': self.defense_score,
            'result': result
        }
    
    def _aggregate(self, updates: List[np.ndarray], defense: str) -> np.ndarray:
        """Aggregate updates using selected defense."""
        if not updates:
            return np.array([])
        
        if defense == 'krum':
            return self._krum(updates)
        elif defense == 'multi_krum':
            return self._multi_krum(updates)
        elif defense == 'trimmed_mean':
            return self._trimmed_mean(updates)
        elif defense == 'median':
            return self._median(updates)
        else:
            return np.mean(updates, axis=0)
    
    def _krum(self, updates: List[np.ndarray]) -> np.ndarray:
        """Krum aggregation."""
        if len(updates) <= 1:
            return updates[0] if updates else np.array([])
        
        n = len(updates)
        distances = []
        
        for i, u1 in enumerate(updates):
            dist_sum = sum([np.linalg.norm(u1 - u2) for j, u2 in enumerate(updates) if i != j])
            distances.append((i, dist_sum))
        
        distances.sort(key=lambda x: x[1])
        return updates[distances[0][0]]
    
    def _multi_krum(self, updates: List[np.ndarray]) -> np.ndarray:
        """Multi-Krum aggregation."""
        if len(updates) <= 1:
            return updates[0] if updates else np.array([])
        
        n = len(updates)
        distances = []
        
        for i, u1 in enumerate(updates):
            dist_sum = sum([np.linalg.norm(u1 - u2) for j, u2 in enumerate(updates) if i != j])
            distances.append((i, dist_sum))
        
        distances.sort(key=lambda x: x[1])
        m = min(3, len(distances))
        selected_indices = [i for i, _ in distances[:m]]
        selected_updates = [updates[i] for i in selected_indices]
        
        return np.mean(selected_updates, axis=0)
    
    def _trimmed_mean(self, updates: List[np.ndarray]) -> np.ndarray:
        """Trimmed mean aggregation."""
        if len(updates) <= 2:
            return np.mean(updates, axis=0)
        
        n = len(updates)
        n_trim = int(n * 0.2)
        
        norms = np.array([np.linalg.norm(u) for u in updates])
        sorted_indices = np.argsort(norms)
        keep_indices = sorted_indices[n_trim:-n_trim]
        
        if len(keep_indices) == 0:
            keep_indices = sorted_indices[:1]
        
        kept_updates = [updates[i] for i in keep_indices]
        return np.mean(kept_updates, axis=0)
    
    def _median(self, updates: List[np.ndarray]) -> np.ndarray:
        """Median aggregation."""
        if len(updates) == 0:
            return np.array([])
        
        updates_np = np.array(updates)
        return np.median(updates_np, axis=0)
    
    def run_game(self, max_rounds: int = 20) -> Dict:
        """Run the full game."""
        print("=== Attack-Defense Game Started ===")
        print(f"Clients: {self.num_clients} (honest: {self.num_clients - self.num_malicious}, malicious: {self.num_malicious})")
        print()
        
        for _ in range(max_rounds):
            if self.game_over:
                break
            
            result = self.play_round()
            
            print(f"Round {result['round']}:")
            print(f"  Defense: {result['defense']}")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Accuracy Drop: {result['accuracy_drop']:.4f}")
            print(f"  Result: {result['result']}")
            print(f"  Score: Attack {result['attack_score']} - {result['defense_score']} Defense")
            print()
            
            # Check game over conditions
            if self.round >= max_rounds:
                self.game_over = True
        
        # Final report
        print("="*50)
        print("          GAME OVER          ")
        print("="*50)
        print(f"Final Score: Attack {self.attack_score} - {self.defense_score} Defense")
        print(f"Total Rounds: {self.round}")
        print(f"Final Accuracy: {self.accuracy_history[-1]:.4f}")
        print("="*50)
        
        return {
            'accuracy_history': self.accuracy_history,
            'defense_history': self.defense_history,
            'attack_params_history': self.attack_params_history,
            'game_log': self.game_log,
            'final_score': {'attack': self.attack_score, 'defense': self.defense_score},
            'defense_stats': self.defense_selector.get_stats()
        }
    
    def visualize_game(self):
        """Print a text-based visualization of the game."""
        print("\n--- GAME VISUALIZATION ---")
        print("Round | Defense   | Accuracy | Result")
        print("------|-----------|----------|--------")
        
        for log in self.game_log:
            acc_str = f"{log['accuracy']*100:.1f}%"
            print(f"{log['round']:5d} | {log['defense']:10s} | {acc_str:9s} | {log['result']}")


def run_game_demo():
    """Run a demonstration of the attack-defense game."""
    coordinator = GameCoordinator(num_clients=20, num_malicious=4)
    results = coordinator.run_game(max_rounds=15)
    coordinator.visualize_game()
    
    return results


if __name__ == "__main__":
    run_game_demo()
