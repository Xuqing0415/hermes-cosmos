"""
Adaptive Defense and Attack-Defense Game Module

Implements adaptive defense mechanisms and attack-defense game simulation.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional


class AdaptiveDefenseServer:
    """
    Defense server that can dynamically switch defense strategies based on detected anomalies.
    
    Supports:
    - Anomaly detection (gradient norm, cosine similarity)
    - Dynamic defense switching (Trimmed Mean → Krum → Multi-Krum)
    - Adaptive threshold adjustment
    """
    
    def __init__(self, initial_defense: str = 'trimmed_mean', anomaly_threshold: float = 3.0):
        """
        Initialize adaptive defense server.
        
        Args:
            initial_defense: Initial defense strategy ('trimmed_mean', 'krum', 'multi_krum')
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.defenses = ['trimmed_mean', 'krum', 'multi_krum']
        self.current_defense = initial_defense
        self.anomaly_threshold = anomaly_threshold
        
        # Defense parameters
        self.trim_ratio = 0.2
        self.krum_k = 1
        self.multi_krum_m = 3  # Number of candidates in Multi-Krum
        
        # Anomaly detection state
        self.norm_history = []
        self.similarity_history = []
        self.anomaly_counts = []
        self.consecutive_anomalies = 0
        
        # Statistics
        self.defense_switches = 0
        self.defense_history = [initial_defense]
    
    def _detect_anomalies(self, updates: List[np.ndarray]) -> Tuple[List[int], float]:
        """
        Detect anomalous updates based on gradient norm and cosine similarity.
        
        Args:
            updates: List of client updates
            
        Returns:
            Tuple of (anomaly_indices, anomaly_ratio)
        """
        if len(updates) <= 1:
            return [], 0.0
        
        # Compute norms
        norms = np.array([np.linalg.norm(u) for u in updates])
        norm_mean = np.mean(norms)
        norm_std = np.std(norms) if len(norms) > 1 else 1.0
        
        # Compute cosine similarity to mean
        mean_update = np.mean(updates, axis=0)
        mean_norm = np.linalg.norm(mean_update)
        
        similarities = []
        for u in updates:
            if mean_norm > 0 and np.linalg.norm(u) > 0:
                sim = np.dot(u.flatten(), mean_update.flatten()) / (np.linalg.norm(u) * mean_norm)
            else:
                sim = 0.0
            similarities.append(sim)
        
        sim_mean = np.mean(similarities)
        sim_std = np.std(similarities) if len(similarities) > 1 else 1.0
        
        # Detect anomalies
        anomaly_indices = []
        for i in range(len(updates)):
            z_norm = abs(norms[i] - norm_mean) / (norm_std + 1e-10)
            z_sim = abs(similarities[i] - sim_mean) / (sim_std + 1e-10)
            
            if z_norm > self.anomaly_threshold or z_sim > self.anomaly_threshold:
                anomaly_indices.append(i)
        
        # Update history
        self.norm_history.append(norm_mean)
        self.similarity_history.append(sim_mean)
        self.anomaly_counts.append(len(anomaly_indices))
        
        return anomaly_indices, len(anomaly_indices) / len(updates)
    
    def _switch_defense(self, new_defense: str):
        """Switch to a new defense strategy."""
        if new_defense != self.current_defense and new_defense in self.defenses:
            print(f"🔄 Switching defense from {self.current_defense} to {new_defense}")
            self.current_defense = new_defense
            self.defense_history.append(new_defense)
            self.defense_switches += 1
            self.consecutive_anomalies = 0  # Reset counter
    
    def _adapt_to_anomalies(self, anomaly_ratio: float):
        """Adapt defense strategy based on anomaly ratio."""
        if anomaly_ratio > 0.3:
            self.consecutive_anomalies += 1
            
            if self.consecutive_anomalies >= 2:
                # Switch to stronger defense
                current_idx = self.defenses.index(self.current_defense)
                if current_idx < len(self.defenses) - 1:
                    self._switch_defense(self.defenses[current_idx + 1])
        else:
            self.consecutive_anomalies = 0
            
            # Can switch back to weaker but more efficient defense
            if self.current_defense != 'trimmed_mean' and len(self.anomaly_counts) > 5:
                recent_anomalies = sum(self.anomaly_counts[-5:]) / 5
                if recent_anomalies < 0.1:
                    self._switch_defense('trimmed_mean')
    
    def aggregate(self, updates: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate updates with adaptive defense.
        
        Args:
            updates: List of client updates
            
        Returns:
            Aggregated update
        """
        if not updates:
            return np.array([])
        
        # Detect anomalies
        anomaly_indices, anomaly_ratio = self._detect_anomalies(updates)
        
        # Adapt defense strategy
        self._adapt_to_anomalies(anomaly_ratio)
        
        # Apply current defense
        if self.current_defense == 'trimmed_mean':
            return self._trimmed_mean(updates)
        elif self.current_defense == 'krum':
            return self._krum(updates)
        elif self.current_defense == 'multi_krum':
            return self._multi_krum(updates)
        else:
            return np.mean(updates, axis=0)
    
    def _trimmed_mean(self, updates: List[np.ndarray]) -> np.ndarray:
        """Trimmed mean aggregation."""
        if len(updates) <= 2:
            return np.mean(updates, axis=0)
        
        n = len(updates)
        n_trim = int(n * self.trim_ratio)
        
        if n_trim == 0:
            return np.mean(updates, axis=0)
        
        norms = np.array([np.linalg.norm(u) for u in updates])
        sorted_indices = np.argsort(norms)
        keep_indices = sorted_indices[n_trim:-n_trim]
        
        if len(keep_indices) == 0:
            keep_indices = sorted_indices[:1]
        
        kept_updates = [updates[i] for i in keep_indices]
        return np.mean(kept_updates, axis=0)
    
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
        selected_idx = distances[0][0]
        return updates[selected_idx]
    
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
        m = min(self.multi_krum_m, len(distances))
        selected_indices = [i for i, _ in distances[:m]]
        selected_updates = [updates[i] for i in selected_indices]
        
        return np.mean(selected_updates, axis=0)
    
    def get_stats(self) -> Dict:
        """Get defense statistics."""
        return {
            'current_defense': self.current_defense,
            'defense_switches': self.defense_switches,
            'defense_history': self.defense_history,
            'consecutive_anomalies': self.consecutive_anomalies,
            'avg_anomaly_ratio': sum(self.anomaly_counts) / len(self.anomaly_counts) if self.anomaly_counts else 0.0
        }


class OnlineAttackEvolution:
    """
    Online attack evolution using genetic algorithm.
    
    Adapts attack parameters based on observed defense effectiveness.
    """
    
    def __init__(self, population_size: int = 4, mutation_rate: float = 0.3):
        """
        Initialize online attack evolution.
        
        Args:
            population_size: Size of attack parameter population
            mutation_rate: Probability of mutation
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Current population of attack parameters
        self.population = [self._random_individual() for _ in range(population_size)]
        
        # History
        self.generation = 0
        self.best_fitness_history = []
        self.best_params_history = []
    
    def _random_individual(self) -> Dict:
        """Generate random attack parameters."""
        return {
            'attack_type': random.choice(['label_flip', 'gradient_scale', 'backdoor']),
            'intensity': random.uniform(1.5, 10.0),
            'mal_ratio': random.uniform(0.1, 0.4),
            'start_round': random.randint(5, 20),
            'trigger_pos': (random.randint(0, 31), random.randint(0, 31)) if random.random() > 0.5 else None
        }
    
    def _evaluate_fitness(self, params: Dict, observed_accuracy: float) -> float:
        """
        Evaluate fitness based on observed accuracy drop.
        
        Args:
            params: Attack parameters
            observed_accuracy: Current model accuracy
            
        Returns:
            Fitness score (higher = better attack)
        """
        # Fitness = 1 - accuracy (lower accuracy = better attack)
        return 1 - observed_accuracy
    
    def evolve(self, observed_accuracy: float):
        """
        Evolve attack parameters based on observed accuracy.
        
        Args:
            observed_accuracy: Current model accuracy after attack
        """
        # Evaluate fitness for each individual
        fitnesses = [self._evaluate_fitness(ind, observed_accuracy) for ind in self.population]
        
        # Selection (tournament selection)
        new_population = []
        for _ in range(self.population_size):
            # Select two parents
            i1 = random.choices(range(self.population_size), weights=fitnesses, k=1)[0]
            i2 = random.choices(range(self.population_size), weights=fitnesses, k=1)[0]
            
            # Crossover
            child = self._crossover(self.population[i1], self.population[i2])
            
            # Mutation
            child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Track best individual
        final_fitnesses = [self._evaluate_fitness(ind, observed_accuracy) for ind in self.population]
        best_idx = np.argmax(final_fitnesses)
        best_fitness = final_fitnesses[best_idx]
        best_params = self.population[best_idx]
        
        self.best_fitness_history.append(best_fitness)
        self.best_params_history.append(best_params)
        
        print(f"🧬 Generation {self.generation}: Best fitness = {best_fitness:.4f}")
        
        return best_params
    
    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        """Crossover two individuals."""
        child = {}
        for key in p1:
            if random.random() > 0.5:
                child[key] = p1[key]
            else:
                child[key] = p2[key]
        return child
    
    def _mutate(self, ind: Dict) -> Dict:
        """Mutate an individual."""
        mutated = ind.copy()
        
        if random.random() < self.mutation_rate:
            mutated['intensity'] += random.uniform(-2.0, 2.0)
            mutated['intensity'] = max(1.0, min(20.0, mutated['intensity']))
        
        if random.random() < self.mutation_rate:
            mutated['mal_ratio'] += random.uniform(-0.1, 0.1)
            mutated['mal_ratio'] = max(0.05, min(0.5, mutated['mal_ratio']))
        
        if random.random() < self.mutation_rate * 0.5:
            mutated['attack_type'] = random.choice(['label_flip', 'gradient_scale', 'backdoor'])
        
        return mutated
    
    def get_best_params(self) -> Dict:
        """Get the best attack parameters."""
        if not self.best_params_history:
            return self.population[0]
        return self.best_params_history[-1]


class AttackDefenseGame:
    """
    Attack-Defense Game Simulator.
    
    Simulates a turn-based game between attacker and defender.
    """
    
    def __init__(self, num_clients: int = 20, model_shape: Tuple[int, int] = (10, 784)):
        """
        Initialize attack-defense game.
        
        Args:
            num_clients: Number of clients
            model_shape: Shape of model parameters
        """
        self.num_clients = num_clients
        self.model_shape = model_shape
        
        # Game state
        self.global_model = np.random.randn(*model_shape) * 0.01
        self.round = 0
        self.game_over = False
        
        # Players
        self.defender = AdaptiveDefenseServer(initial_defense='trimmed_mean')
        self.attacker = OnlineAttackEvolution(population_size=4)
        
        # Statistics
        self.accuracy_history = []
        self.defense_history = []
        self.attack_params_history = []
        self.winner = None
        
        # Score tracking
        self.attack_score = 0
        self.defense_score = 0
        self.round_results = []
    
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
    
    def _generate_updates(self, attack_params: Dict) -> List[np.ndarray]:
        """Generate client updates with possible attacks."""
        num_malicious = int(self.num_clients * attack_params.get('mal_ratio', 0.2))
        attack_type = attack_params.get('attack_type', 'label_flip')
        intensity = attack_params.get('intensity', 5.0)
        
        updates = []
        for i in range(self.num_clients):
            if i < num_malicious:
                # Malicious update
                if attack_type == 'gradient_scale':
                    update = np.random.randn(*self.model_shape) * intensity * 0.1
                elif attack_type == 'label_flip':
                    # Flip labels would produce adversarial gradients
                    update = np.random.randn(*self.model_shape) * 0.5
                else:  # backdoor
                    update = np.random.randn(*self.model_shape) * 0.3
            else:
                # Honest update (small random gradient)
                update = np.random.randn(*self.model_shape) * 0.01
            
            updates.append(update)
        
        return updates
    
    def play_round(self) -> Dict:
        """Play one round of the game."""
        self.round += 1
        
        # Attacker evolves based on previous round's accuracy
        if self.round > 1:
            prev_accuracy = self.accuracy_history[-1]
            attack_params = self.attacker.evolve(prev_accuracy)
        else:
            attack_params = self.attacker.get_best_params()
        
        # Generate updates with current attack
        updates = self._generate_updates(attack_params)
        
        # Defender aggregates with adaptive defense
        aggregated_update = self.defender.aggregate(updates)
        
        # Update global model
        self.global_model += aggregated_update * 0.01
        
        # Evaluate
        accuracy = self._evaluate()
        self.accuracy_history.append(accuracy)
        self.defense_history.append(self.defender.current_defense)
        self.attack_params_history.append(attack_params)
        
        # Determine round winner
        if self.round > 1:
            prev_acc = self.accuracy_history[-2]
            acc_change = accuracy - prev_acc
            
            if acc_change < -0.05:
                # Attack successful
                self.attack_score += 1
                result = 'ATTACK WIN'
            elif acc_change > 0.05:
                # Defense successful
                self.defense_score += 1
                result = 'DEFENSE WIN'
            else:
                result = 'DRAW'
        else:
            result = 'INITIAL'
        
        self.round_results.append(result)
        
        # Check game over conditions
        if self.round >= 20:
            self.game_over = True
            if self.attack_score > self.defense_score:
                self.winner = 'ATTACKER'
            elif self.defense_score > self.attack_score:
                self.winner = 'DEFENDER'
            else:
                self.winner = 'DRAW'
        
        return {
            'round': self.round,
            'accuracy': accuracy,
            'defense': self.defender.current_defense,
            'attack_type': attack_params['attack_type'],
            'result': result,
            'attack_score': self.attack_score,
            'defense_score': self.defense_score
        }
    
    def run_game(self, max_rounds: int = 20) -> Dict:
        """Run the full game."""
        print("=== ⚔️ Attack-Defense Game Started ⚔️ ===")
        
        for _ in range(max_rounds):
            if self.game_over:
                break
            
            result = self.play_round()
            
            print(f"\n🔄 Round {result['round']}")
            print(f"   Defense: {result['defense']}")
            print(f"   Attack: {result['attack_type']} (intensity={result.get('attack_params', {}).get('intensity', 0):.2f})")
            print(f"   Accuracy: {result['accuracy']:.4f}")
            print(f"   Result: {result['result']}")
            print(f"   Score: Attack {result['attack_score']} - {result['defense_score']} Defense")
        
        # Final report
        print("\n" + "="*50)
        print("          🏆 GAME OVER 🏆          ")
        print("="*50)
        print(f"Final Score: Attack {self.attack_score} - {self.defense_score} Defense")
        print(f"Winner: {self.winner}")
        print(f"Total Rounds: {self.round}")
        print(f"Defense Switches: {self.defender.defense_switches}")
        print("="*50)
        
        return {
            'accuracy_history': self.accuracy_history,
            'defense_history': self.defense_history,
            'attack_params_history': self.attack_params_history,
            'round_results': self.round_results,
            'final_score': {'attack': self.attack_score, 'defense': self.defense_score},
            'winner': self.winner
        }


def run_game_demo():
    """Run a demonstration of the attack-defense game."""
    game = AttackDefenseGame(num_clients=20, model_shape=(10, 784))
    results = game.run_game(max_rounds=15)
    
    # Print summary statistics
    print("\n📊 GAME STATISTICS")
    print(f"Initial Accuracy: {results['accuracy_history'][0]:.4f}")
    print(f"Final Accuracy: {results['accuracy_history'][-1]:.4f}")
    print(f"Accuracy Change: {(results['accuracy_history'][-1] - results['accuracy_history'][0])*100:.2f}%")
    
    # Print round results
    print("\n🔢 ROUND RESULTS:")
    for i, result in enumerate(results['round_results'], 1):
        print(f"  Round {i}: {result}")


if __name__ == "__main__":
    run_game_demo()
