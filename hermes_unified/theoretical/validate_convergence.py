"""
Convergence Bound Validation Script

Runs simulations to validate the theoretical convergence bound for Byzantine-tolerant
federated learning.
"""

import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.federated_simulator import FederatedSimulator


def run_convergence_experiment():
    """Run convergence experiment for different malicious client ratios."""
    print("=== Convergence Bound Validation ===")
    print("Running experiments with different malicious client ratios...")
    
    # Parameters
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]
    num_clients = 50
    num_rounds = 100
    local_epochs = 2
    
    results = {
        'alphas': alphas,
        'final_accuracy': [],
        'loss_history': [],
        'accuracy_history': []
    }
    
    for alpha in alphas:
        print(f"\n--- Testing malicious ratio α = {alpha} ---")
        
        # Create attack configuration
        attack_config = {
            'attack_type': 'gradient_scale',
            'intensity': 5.0,
            'mal_ratio': alpha,
            'start_round': 0
        }
        
        # Create simulator with Krum defense
        sim = FederatedSimulator(
            num_clients=num_clients,
            model_shape=(10, 784),
            num_features=784,
            num_classes=10,
            attack_config=attack_config,
            defense_type='krum'
        )
        
        # Setup clients with Non-IID data
        sim.setup_clients(data_distribution='non_iid', alpha=0.5)
        
        # Run simulation
        result = sim.run_federated_training(
            num_rounds=num_rounds,
            clients_per_round=10,
            local_epochs=local_epochs,
            use_fedprox=False
        )
        
        # Store results
        final_acc = result['accuracy_history'][-1]
        results['final_accuracy'].append(final_acc)
        results['loss_history'].append(result['loss_history'])
        results['accuracy_history'].append(result['accuracy_history'])
        
        print(f"Final Accuracy: {final_acc:.4f}")
        print(f"Error (1-acc): {1 - final_acc:.4f}")
    
    return results


def fit_quadratic_model(alphas, errors):
    """Fit a quadratic model: error = c * alpha^2 + baseline"""
    # Convert to numpy arrays
    alphas = np.array(alphas)
    errors = np.array(errors)
    
    # Quadratic fitting: y = a*x^2 + b
    # We can use linear regression by letting x' = x^2
    x_squared = alphas ** 2
    
    # Solve for a and b: errors = a * x_squared + b
    X = np.column_stack([x_squared, np.ones_like(x_squared)])
    coeffs, _, _, _ = np.linalg.lstsq(X, errors, rcond=None)
    
    a, b = coeffs
    
    # Calculate R-squared
    y_pred = a * x_squared + b
    ss_tot = np.sum((errors - np.mean(errors)) ** 2)
    ss_res = np.sum((errors - y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return a, b, r_squared


def generate_report(results):
    """Generate analysis report."""
    print("\n" + "="*70)
    print("          CONVERGENCE BOUND VALIDATION REPORT          ")
    print("="*70)
    
    # Extract data
    alphas = results['alphas']
    final_acc = results['final_accuracy']
    errors = [1 - acc for acc in final_acc]
    alphas_squared = [a**2 for a in alphas]
    
    # Fit quadratic model
    a, b, r_squared = fit_quadratic_model(alphas, errors)
    
    # Print table
    print("\nSIMULATION RESULTS")
    print("-" * 60)
    print(f"{'alpha':<8} {'alpha2':<8} {'Accuracy':<12} {'Error (1-acc)':<15} {'Predicted':<12}")
    print("-" * 60)
    
    for i, (alpha, alpha_sq, acc, error) in enumerate(zip(alphas, alphas_squared, final_acc, errors)):
        predicted = a * alpha_sq + b
        print(f"{alpha:<8.1f} {alpha_sq:<8.2f} {acc:<12.4f} {error:<15.4f} {predicted:<12.4f}")
    
    print("-" * 60)
    
    # Print fitting results
    print(f"\nQUADRATIC FIT RESULTS")
    print(f"Model: error = {a:.4f} * alpha^2 + {b:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    
    # Interpret results
    print(f"\nINTERPRETATION")
    if r_squared > 0.95:
        print(f"Excellent fit! R-squared = {r_squared:.4f}")
        print(f"   The error is well approximated by a quadratic function of alpha.")
        print(f"   This confirms the theoretical prediction: error ~ alpha^2")
    elif r_squared > 0.8:
        print(f"Good fit. R-squared = {r_squared:.4f}")
        print(f"   There is some deviation from the theoretical prediction.")
    else:
        print(f"Poor fit. R-squared = {r_squared:.4f}")
        print(f"   The simulation results do not match the quadratic model.")
    
    # Compare with theoretical bound
    print(f"\nTHEORETICAL COMPARISON")
    print(f"Baseline error (alpha=0): {errors[0]:.4f}")
    print(f"Maximum error (alpha=0.4): {errors[-1]:.4f}")
    print(f"Error increase factor: {(errors[-1] - errors[0]) / errors[0]:.2f}x")
    print(f"Theoretical prediction: alpha^2 increase = {0.4**2 / 0.1**2:.0f}x (for alpha=0.4 vs alpha=0.1)")
    
    print("\n" + "="*70)
    
    # Save results
    save_results(alphas, final_acc, errors, a, b, r_squared)


def save_results(alphas, accuracies, errors, a, b, r_squared):
    """Save results to CSV file."""
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    print(f"Creating results directory at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical results
    with open(os.path.join(output_dir, 'convergence_results.csv'), 'w') as f:
        f.write('alpha,accuracy,error,alpha_squared,predicted\n')
        for i, (alpha, acc, error) in enumerate(zip(alphas, accuracies, errors)):
            predicted = a * (alpha**2) + b
            f.write(f'{alpha},{acc:.6f},{error:.6f},{alpha**2:.4f},{predicted:.6f}\n')
    
    # Save fitting parameters
    with open(os.path.join(output_dir, 'fitting_parameters.txt'), 'w') as f:
        f.write(f'Quadratic Fit Parameters:\n')
        f.write(f'  Coefficient (a): {a:.6f}\n')
        f.write(f'  Baseline (b): {b:.6f}\n')
        f.write(f'  R-squared: {r_squared:.6f}\n')
        f.write(f'\nModel: error = {a:.4f} * alpha^2 + {b:.4f}\n')
    
    print(f"\nResults saved to {output_dir}")


def main():
    """Main function."""
    # Run experiment
    results = run_convergence_experiment()
    
    # Generate report
    generate_report(results)
    
    print("\nConvergence validation complete!")
    print("Check the generated report and results files for detailed analysis.")


if __name__ == "__main__":
    main()
