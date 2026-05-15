"""
Federated Learning Attack-Defense Simulator

Simulates attack and defense scenarios in federated learning.
"""

import numpy as np
import random
from typing import List, Dict, Tuple
import time

from .federated_simulator import FLClient, NonIIDDataGenerator
from .attacks import AttackClient, AttackType
from .defenses import DefenseServer, DefenseType


class AttackDefenseSimulator:
    """
    Simulator for attack-defense scenarios in federated learning.
    
    Combines malicious clients with defense mechanisms to study security.
    """
    
    def __init__(self, num_clients: int = 20, num_malicious: int = 3,
                 model_shape: Tuple[int, int] = (10, 784)):
        """
        Initialize attack-defense simulator.
        
        Args:
            num_clients: Total number of clients
            num_malicious: Number of malicious clients
            model_shape: Shape of model parameters
        """
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.model_shape = model_shape
        
        # Initialize global model
        self.global_model = np.random.randn(*model_shape) * 0.01
        
        # Clients
        self.honest_clients: List[FLClient] = []
        self.malicious_clients: List[AttackClient] = []
        
        # Defense server
        self.defense_server = None
        
        # Statistics
        self.loss_history = []
        self.accuracy_history = []
        self.attack_events = []
        self.defense_events = []
        
        # Battle state
        self.battle_active = False
        self.attack_start_round = None
    
    def setup_clients(self, attack_type: str = 'label_flip', attack_strength: float = 0.5):
        """
        Setup honest and malicious clients.
        
        Args:
            attack_type: Type of attack for malicious clients
            attack_strength: Attack strength (0-1)
        """
        # Generate synthetic data
        num_features = self.model_shape[1]
        num_classes = self.model_shape[0]
        
        X = np.random.randn(10000, num_features)
        true_weights = np.random.randn(*self.model_shape)
        logits = X @ true_weights.T
        y = np.argmax(logits, axis=1)
        
        # Partition data
        data_generator = NonIIDDataGenerator(num_classes=num_classes)
        client_data = data_generator.generate_dirichlet_partition(X, y, self.num_clients, alpha=0.5)
        
        # Assign clients
        honest_indices = random.sample(range(self.num_clients), self.num_clients - self.num_malicious)
        malicious_indices = [i for i in range(self.num_clients) if i not in honest_indices]
        
        # Create honest clients
        for idx in honest_indices:
            client = FLClient(
                client_id=idx,
                data=client_data[idx],
                model_shape=self.model_shape
            )
            self.honest_clients.append(client)
        
        # Create malicious clients
        attack_types = ['label_flip', 'gradient_scale', 'backdoor', 'adaptive']
        for idx in malicious_indices:
            atype = attack_type if attack_type != 'mixed' else random.choice(attack_types)
            malicious_client = AttackClient(
                client_id=idx,
                data=client_data[idx],
                model_shape=self.model_shape,
                attack_type=atype,
                attack_strength=attack_strength
            )
            self.malicious_clients.append(malicious_client)
        
        print(f"🎭 Setup complete: {len(self.honest_clients)} honest clients, {len(self.malicious_clients)} malicious clients")
        print(f"⚔️ Attack type: {attack_type}")
    
    def set_defense(self, defense_type: str = 'trimmed_mean', **kwargs):
        """
        Set defense mechanism.
        
        Args:
            defense_type: Type of defense to use
            kwargs: Additional defense parameters
        """
        self.defense_server = DefenseServer(defense_type, **kwargs)
        
        # For FLTrust, set up trusted data
        if defense_type == DefenseType.FLTRUST:
            X_trust = np.random.randn(500, self.model_shape[1])
            y_trust = np.argmax(X_trust @ np.random.randn(*self.model_shape).T, axis=1)
            self.defense_server.set_trusted_data(X_trust, y_trust)
        
        print(f"🛡️ Defense enabled: {defense_type}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate global model."""
        pred = X_test @ self.global_model.T
        predictions = np.argmax(pred, axis=1)
        
        accuracy = np.mean(predictions == y_test)
        loss = np.mean((pred - np.eye(self.model_shape[0])[y_test]) ** 2)
        
        return loss, accuracy
    
    def run_simulation(self, num_rounds: int = 50, clients_per_round: int = 5,
                       local_epochs: int = 1, attack_start_round: int = 10):
        """
        Run attack-defense simulation.
        
        Args:
            num_rounds: Number of federated rounds
            clients_per_round: Number of clients selected per round
            local_epochs: Number of local epochs
            attack_start_round: Round when attacks begin
        """
        print("\n🚀 Starting Attack-Defense Simulation...")
        
        # Generate test data
        X_test = np.random.randn(1000, self.model_shape[1])
        y_test = np.argmax(X_test @ np.random.randn(*self.model_shape).T, axis=1)
        
        self.attack_start_round = attack_start_round
        
        for round_idx in range(num_rounds):
            # Check if attack should start
            is_attack_round = round_idx >= attack_start_round and self.malicious_clients
            
            if is_attack_round and not self.battle_active:
                self.battle_active = True
                print(f"\n🔥 ATTACK PHASE STARTED at Round {round_idx}!")
            
            # Select clients
            all_clients = self.honest_clients.copy()
            
            # Add malicious clients if attack round
            if is_attack_round:
                all_clients.extend(self.malicious_clients)
            
            # Random selection
            selected = random.sample(all_clients, min(clients_per_round, len(all_clients)))
            
            # Collect updates
            updates = []
            was_attack = False
            
            for client in selected:
                if hasattr(client, 'is_malicious') and client.is_malicious:
                    update = client.train_local(self.global_model, num_epochs=local_epochs)
                    was_attack = True
                else:
                    update = client.train_local(self.global_model, num_epochs=local_epochs)
                
                updates.append(update)
            
            # Apply defense
            if self.defense_server:
                aggregated_update = self.defense_server.defend(updates, self.global_model)
            else:
                aggregated_update = np.mean(updates, axis=0)
            
            # Update global model
            self.global_model += aggregated_update
            
            # Evaluate
            loss, accuracy = self.evaluate(X_test, y_test)
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            # Check for attack impact
            if was_attack and round_idx > 0:
                prev_acc = self.accuracy_history[-2]
                acc_drop = (prev_acc - accuracy) * 100
                
                if acc_drop > 5:  # Significant drop indicates successful attack
                    self.attack_events.append({
                        'round': round_idx,
                        'accuracy_drop': acc_drop,
                        'attackers': [c.client_id for c in self.malicious_clients]
                    })
                    print(f"💀 ATTACK SUCCESSFUL at Round {round_idx}! Accuracy dropped by {acc_drop:.2f}%")
            
            # Print progress
            if round_idx % 10 == 0:
                status = "⚔️ ATTACK" if is_attack_round else "🔒 SAFE"
                print(f"Round {round_idx} [{status}]: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        print("\n🏁 Simulation complete!")
        self._generate_battle_report()
    
    def _generate_battle_report(self):
        """Generate a comprehensive battle report."""
        print("\n" + "="*60)
        print("          📊 FEDERATED LEARNING BATTLE REPORT          ")
        print("="*60)
        
        # Attack summary
        print("\n⚔️ ATTACK SUMMARY")
        print(f"Total malicious clients: {len(self.malicious_clients)}")
        attack_types = {}
        for c in self.malicious_clients:
            atype = c.attack_type
            attack_types[atype] = attack_types.get(atype, 0) + 1
        
        for atype, count in attack_types.items():
            print(f"  - {atype}: {count} client(s)")
        
        print(f"Successful attacks: {len(self.attack_events)}")
        
        # Defense summary
        print("\n🛡️ DEFENSE SUMMARY")
        if self.defense_server:
            defense_summary = self.defense_server.get_defense_summary()
            print(f"Defense type: {defense_summary['defense_type']}")
            print(f"Updates filtered: {defense_summary['total_updates_filtered']}")
        else:
            print("No defense enabled")
        
        # Final accuracy
        final_acc = self.accuracy_history[-1] if self.accuracy_history else 0
        peak_acc = max(self.accuracy_history) if self.accuracy_history else 0
        acc_drop = (peak_acc - final_acc) * 100
        
        print("\n📈 PERFORMANCE METRICS")
        print(f"Final Accuracy: {final_acc:.4f}")
        print(f"Peak Accuracy: {peak_acc:.4f}")
        print(f"Total Accuracy Drop: {acc_drop:.2f}%")
        
        # Battle outcome
        print("\n🏆 BATTLE OUTCOME")
        if acc_drop < 10:
            print("VICTORY! 🎉 Defense successfully mitigated attacks!")
        elif acc_drop < 30:
            print("STALEMATE ⚔️ Attacks caused some damage but defense held!")
        else:
            print("DEFEAT 💀 Attacks overwhelmed the defense!")
        
        print("="*60)
    
    def get_results(self) -> Dict:
        """Get simulation results."""
        return {
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history,
            'attack_events': self.attack_events,
            'defense_events': self.defense_events,
            'num_honest_clients': len(self.honest_clients),
            'num_malicious_clients': len(self.malicious_clients),
            'defense_type': self.defense_server.defense_type if self.defense_server else None
        }


def run_attack_defense_simulation():
    """Run a complete attack-defense simulation."""
    print("=== 🔮 Federated Learning Attack-Defense Simulation 🔮 ===")
    
    # Create simulator
    sim = AttackDefenseSimulator(
        num_clients=20,
        num_malicious=4,
        model_shape=(10, 784)
    )
    
    # Setup clients with mixed attacks
    sim.setup_clients(attack_type='mixed', attack_strength=0.7)
    
    # Enable defense
    sim.set_defense(defense_type='anomaly_detection', anomaly_threshold=2.5)
    
    # Run simulation
    sim.run_simulation(
        num_rounds=50,
        clients_per_round=6,
        local_epochs=2,
        attack_start_round=15
    )
    
    return sim.get_results()


if __name__ == "__main__":
    results = run_attack_defense_simulation()
