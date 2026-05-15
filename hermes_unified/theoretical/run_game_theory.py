"""
Run Game Theory Simulation

Demonstrates the adaptive defense vs adaptive attack game in federated learning.
"""

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.federated_coordinator import GameCoordinator, AdaptiveDefenseSelector, AdaptiveAttackClient


def plot_results(results, output_dir='figures'):
    """Generate visualization of game results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        rounds = np.arange(1, len(results['accuracy_history']) + 1)
        accuracies = np.array(results['accuracy_history'])
        defenses = results['defense_history']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(rounds, accuracies, 'b-', linewidth=2, label='Accuracy')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Evolution During Attack-Defense Game')
        ax1.grid(True)
        ax1.legend()
        
        for i, defense in enumerate(defenses):
            ax1.text(i + 1, accuracies[i], defense, 
                     ha='center', va='bottom', rotation=45, fontsize=8)
        
        attack_scales = []
        for params in results['attack_params_history']:
            scales = [p['scale'] for p in params]
            attack_scales.append(np.mean(scales))
        
        ax2.plot(rounds, attack_scales, 'r-', linewidth=2, label='Avg Attack Scale')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Attack Scale')
        ax2.set_title('Attack Intensity Evolution')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'game_convergence.png'))
        plt.close()
        
        print(f"Plot saved to {os.path.join(output_dir, 'game_convergence.png')}")
        
    except ImportError:
        print("Matplotlib not installed, skipping plot generation.")


def print_game_summary(results):
    """Print detailed game summary."""
    print("\n" + "="*60)
    print("          GAME THEORY ANALYSIS SUMMARY          ")
    print("="*60)
    
    print("\n1. FINAL SCORES")
    print("-" * 30)
    print(f"   Attack Score: {results['final_score']['attack']}")
    print(f"   Defense Score: {results['final_score']['defense']}")
    
    winner = "Attackers" if results['final_score']['attack'] > results['final_score']['defense'] else \
             "Defenders" if results['final_score']['defense'] > results['final_score']['attack'] else "Draw"
    print(f"   Winner: {winner}")
    
    print("\n2. ACCURACY STATISTICS")
    print("-" * 30)
    acc_history = results['accuracy_history']
    print(f"   Initial Accuracy: {acc_history[0]:.4f}")
    print(f"   Final Accuracy: {acc_history[-1]:.4f}")
    print(f"   Maximum Accuracy: {max(acc_history):.4f}")
    print(f"   Minimum Accuracy: {min(acc_history):.4f}")
    print(f"   Average Accuracy: {np.mean(acc_history):.4f}")
    
    print("\n3. DEFENSE USAGE")
    print("-" * 30)
    defenses = results['defense_history']
    unique_defenses, counts = np.unique(defenses, return_counts=True)
    for d, c in zip(unique_defenses, counts):
        percentage = (c / len(defenses)) * 100
        print(f"   {d}: {c} times ({percentage:.1f}%)")
    
    print("\n4. ATTACK PARAMETER EVOLUTION")
    print("-" * 30)
    scales = []
    for params in results['attack_params_history']:
        scales.extend([p['scale'] for p in params])
    
    print(f"   Initial Attack Scale: {scales[0]:.2f}")
    print(f"   Final Attack Scale: {scales[-1]:.2f}")
    print(f"   Maximum Attack Scale: {max(scales):.2f}")
    print(f"   Minimum Attack Scale: {min(scales):.2f}")
    
    print("\n5. GAME LOG (Key Events)")
    print("-" * 30)
    for log in results['game_log']:
        if log['result'] != 'DRAW':
            print(f"   Round {log['round']}: {log['result']} (Accuracy: {log['accuracy']:.4f})")
    
    print("\n" + "="*60)


def run_multiple_games(num_games=5):
    """Run multiple games to study statistical behavior."""
    print(f"Running {num_games} games for statistical analysis...")
    
    all_results = []
    attack_wins = 0
    defense_wins = 0
    draws = 0
    
    for i in range(num_games):
        print(f"\n--- Game {i+1}/{num_games} ---")
        coordinator = GameCoordinator(num_clients=20, num_malicious=4)
        result = coordinator.run_game(max_rounds=15)
        all_results.append(result)
        
        a_score = result['final_score']['attack']
        d_score = result['final_score']['defense']
        
        if a_score > d_score:
            attack_wins += 1
        elif d_score > a_score:
            defense_wins += 1
        else:
            draws += 1
    
    print("\n" + "="*60)
    print("       STATISTICAL SUMMARY OVER MULTIPLE GAMES       ")
    print("="*60)
    print(f"Total Games: {num_games}")
    print(f"Attack Wins: {attack_wins} ({attack_wins/num_games*100:.1f}%)")
    print(f"Defense Wins: {defense_wins} ({defense_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    avg_final_acc = np.mean([r['accuracy_history'][-1] for r in all_results])
    print(f"\nAverage Final Accuracy: {avg_final_acc:.4f}")
    
    return all_results


def analyze_game_dynamics(results):
    """Analyze game dynamics and equilibrium."""
    print("\n" + "="*60)
    print("          GAME DYNAMICS ANALYSIS          ")
    print("="*60)
    
    rounds = np.arange(1, len(results['accuracy_history']) + 1)
    accuracies = np.array(results['accuracy_history'])
    defenses = results['defense_history']
    
    attack_scales = []
    for params in results['attack_params_history']:
        attack_scales.append(np.mean([p['scale'] for p in params]))
    
    print("\n1. STRATEGIC EVOLUTION")
    print("-" * 30)
    
    defense_changes = []
    prev_defense = defenses[0]
    for i, d in enumerate(defenses):
        if d != prev_defense:
            defense_changes.append((i+1, prev_defense, d))
            prev_defense = d
    
    if defense_changes:
        print("   Defense switches:")
        for round_num, from_def, to_def in defense_changes:
            print(f"     Round {round_num}: {from_def} -> {to_def}")
    else:
        print("   No defense switches occurred")
    
    print("\n2. ATTACK INTENSITY ANALYSIS")
    print("-" * 30)
    scale_changes = np.diff(attack_scales)
    avg_change = np.mean(scale_changes)
    max_change = np.max(np.abs(scale_changes))
    print(f"   Average scale change per round: {avg_change:.3f}")
    print(f"   Maximum scale change: {max_change:.3f}")
    print(f"   Scale volatility: {np.std(scale_changes):.3f}")
    
    print("\n3. EQUILIBRIUM ANALYSIS")
    print("-" * 30)
    
    final_rounds = max(5, len(rounds) // 3)
    late_acc_mean = np.mean(accuracies[-final_rounds:])
    late_acc_std = np.std(accuracies[-final_rounds:])
    late_scale_mean = np.mean(attack_scales[-final_rounds:])
    late_scale_std = np.std(attack_scales[-final_rounds:])
    
    print(f"   Final {final_rounds} rounds analysis:")
    print(f"     Accuracy mean: {late_acc_mean:.4f}")
    print(f"     Accuracy std: {late_acc_std:.4f}")
    print(f"     Attack scale mean: {late_scale_mean:.2f}")
    print(f"     Attack scale std: {late_scale_std:.2f}")
    
    if late_acc_std < 0.01 and late_scale_std < 0.5:
        print("\n     Game appears to have reached equilibrium")
    else:
        print("\n     Game still evolving (no clear equilibrium yet)")
    
    print("\n" + "="*60)


def main():
    """Main function to run game theory experiments."""
    print("=== Federated Learning Attack-Defense Game ===")
    print("Adaptive Defense vs Adaptive Attack")
    print("="*60)
    
    # Single game demonstration
    print("\n--- Running Single Game Demonstration ---")
    coordinator = GameCoordinator(num_clients=20, num_malicious=4)
    results = coordinator.run_game(max_rounds=15)
    coordinator.visualize_game()
    
    # Generate visualization
    plot_results(results)
    
    # Print detailed summary
    print_game_summary(results)
    
    # Analyze game dynamics
    analyze_game_dynamics(results)
    
    # Multiple games for statistics
    print("\n--- Running Multiple Games for Statistical Analysis ---")
    run_multiple_games(num_games=3)
    
    print("\n--- Game Theory Analysis Complete ---")


if __name__ == "__main__":
    main()
