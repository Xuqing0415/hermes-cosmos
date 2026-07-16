"""
LLM Federated Fine-Tuning Runner

Main script to run federated LLM fine-tuning experiments and compare with centralized training.
"""

import os
import sys
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_federated_finetune(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                           num_clients: int = 5,
                           num_rounds: int = 10,
                           clients_per_round: int = 3,
                           local_epochs: int = 1,
                           non_iid_alpha: float = 0.5,
                           samples_per_client: int = 50,
                           use_quantization: bool = False,
                           aggregation_method: str = 'fedavg'):
    """
    Run federated LLM fine-tuning.
    
    Args:
        model_name: HuggingFace model name
        num_clients: Number of federated clients
        num_rounds: Number of federated rounds
        clients_per_round: Number of clients per round
        local_epochs: Number of local epochs
        non_iid_alpha: Non-IIDness parameter (lower = more non-IID)
        samples_per_client: Number of samples per client
        use_quantization: Whether to use 4-bit quantization
        aggregation_method: Aggregation method
    
    Returns:
        results: Dictionary of results
    """
    from llm_model import load_llm, wrap_with_lora
    from non_iid_data import create_synthetic_non_iid_data, tokenize_dataset
    from llm_server import LLMFederatedCoordinator
    
    print("="*60)
    print("Running Federated LLM Fine-Tuning")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Clients: {num_clients}, Rounds: {num_rounds}")
    print(f"Non-IID Alpha: {non_iid_alpha}")
    print("="*60)
    
    # Load model
    model, tokenizer = load_llm(model_name, use_quantization=use_quantization)
    model, _ = wrap_with_lora(model)
    
    # Create Non-IID data
    client_datasets = create_synthetic_non_iid_data(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        alpha=non_iid_alpha
    )
    
    # Tokenize datasets
    tokenized_datasets = [tokenize_dataset(ds, tokenizer) for ds in client_datasets]
    
    # Create test dataset (combine all client data for testing)
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets(client_datasets)
    test_dataset = tokenize_dataset(combined_dataset, tokenizer)
    
    # Create coordinator and run
    coordinator = LLMFederatedCoordinator(model, tokenizer, tokenized_datasets, test_dataset)
    results = coordinator.run(
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        aggregation_method=aggregation_method
    )
    
    # Store additional info
    results['model_name'] = model_name
    results['num_clients'] = num_clients
    results['non_iid_alpha'] = non_iid_alpha
    results['aggregation_method'] = aggregation_method
    
    return results


def run_centralized_finetune(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                             num_epochs: int = 3,
                             batch_size: int = 4,
                             lr: float = 2e-4,
                             use_quantization: bool = False):
    """
    Run centralized LLM fine-tuning for comparison.
    
    Args:
        model_name: HuggingFace model name
        num_epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        use_quantization: Whether to use 4-bit quantization
    
    Returns:
        results: Dictionary of results
    """
    import torch
    from torch.utils.data import DataLoader
    from llm_model import load_llm, wrap_with_lora
    from non_iid_data import create_synthetic_non_iid_data, tokenize_dataset
    from datasets import concatenate_datasets
    
    print("\n" + "="*60)
    print("Running Centralized LLM Fine-Tuning")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}")
    print("="*60)
    
    # Load model
    model, tokenizer = load_llm(model_name, use_quantization=use_quantization)
    model, _ = wrap_with_lora(model)
    device = next(model.parameters()).device
    
    # Create data (same as federated)
    client_datasets = create_synthetic_non_iid_data(
        num_clients=5,
        samples_per_client=50,
        alpha=0.5
    )
    
    # Combine all data for centralized training
    combined_dataset = concatenate_datasets(client_datasets)
    tokenized_dataset = tokenize_dataset(combined_dataset, tokenizer)
    
    # Create dataloader
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training
    model.train()
    loss_history = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids']
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    # Final evaluation
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids']
            )
            
            total_loss += outputs.loss.item() * batch['input_ids'].shape[0]
            total_tokens += batch['input_ids'].numel()
    
    final_perplexity = np.exp(total_loss / total_tokens)
    
    results = {
        'loss_history': loss_history,
        'perplexity_history': [np.exp(l) for l in loss_history],
        'final_perplexity': final_perplexity,
        'model_name': model_name,
        'num_epochs': num_epochs
    }
    
    print(f"\nFinal Perplexity: {final_perplexity:.2f}")
    
    return results


def plot_results(fed_results, centralized_results, output_dir: str = 'figures'):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        rounds = np.arange(1, len(fed_results['perplexity_history']) + 1)
        fed_perplexity = fed_results['perplexity_history']
        
        # Centralized results - adjust for comparison
        centralized_perplexity = centralized_results['perplexity_history']
        centralized_rounds = np.linspace(1, len(fed_results['perplexity_history']), len(centralized_perplexity))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Perplexity comparison
        ax1.plot(rounds, fed_perplexity, 'b-', linewidth=2, label='Federated (LoRA)')
        ax1.plot(centralized_rounds, centralized_perplexity, 'r--', linewidth=2, label='Centralized (LoRA)')
        ax1.set_xlabel('Round/Epoch')
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Perplexity Comparison: Federated vs Centralized')
        ax1.grid(True)
        ax1.legend()
        
        # Communication plot
        ax2.plot(rounds, fed_results['communication_history'], 'g-', linewidth=2, label='Total Communication')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Communication (MB)')
        ax2.set_title('Cumulative Communication Over Time')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'llm_finetune_comparison.png'))
        plt.close()
        
        print(f"\nPlot saved to {os.path.join(output_dir, 'llm_finetune_comparison.png')}")
        
    except ImportError:
        print("\nMatplotlib not installed, skipping plot generation.")


def save_results(fed_results, centralized_results, output_dir: str = 'results'):
    """Save results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save federated results
    fed_path = os.path.join(output_dir, 'federated_results.json')
    with open(fed_path, 'w') as f:
        json.dump({k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in fed_results.items()}, f, indent=2)
    
    # Save centralized results
    centralized_path = os.path.join(output_dir, 'centralized_results.json')
    with open(centralized_path, 'w') as f:
        json.dump({k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in centralized_results.items()}, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")


def print_comparison_summary(fed_results, centralized_results):
    """Print comparison summary."""
    print("\n" + "="*60)
    print("         COMPARISON SUMMARY         ")
    print("="*60)
    
    print("\n1. PERPLEXITY")
    print("-" * 30)
    print(f"Federated Final Perplexity: {fed_results['perplexity_history'][-1]:.2f}")
    print(f"Centralized Final Perplexity: {centralized_results['final_perplexity']:.2f}")
    
    diff = abs(fed_results['perplexity_history'][-1] - centralized_results['final_perplexity'])
    print(f"Difference: {diff:.2f}")
    
    if fed_results['perplexity_history'][-1] < centralized_results['final_perplexity'] * 1.05:
        print(" Federated performance is within 5% of centralized!")
    else:
        print("  Federated performance is significantly worse than centralized")
    
    print("\n2. COMMUNICATION")
    print("-" * 30)
    print(f"Total Communication (Federated): {fed_results['communication_history'][-1]:.2f} MB")
    print(f"Full Model Size (Estimated): ~{(7 * 4):.0f} GB (7B model)")
    print(f"LoRA Size Reduction: {(fed_results['communication_history'][-1] / (7 * 1024)) * 100:.4f}% of full model")
    
    print("\n3. TRAINING DETAILS")
    print("-" * 30)
    print(f"Federated: {fed_results['num_clients']} clients, {len(fed_results['perplexity_history'])} rounds")
    print(f"Centralized: {centralized_results['num_epochs']} epochs")
    
    print("\n" + "="*60)


def main():
    """Main function to run LLM federated fine-tuning experiments."""
    print("=== LLM Federated Fine-Tuning Experiment ===")
    print("Comparing Federated vs Centralized LoRA Fine-Tuning")
    print("="*60)
    
    # Run federated fine-tuning
    fed_results = run_federated_finetune(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_clients=5,
        num_rounds=5,
        clients_per_round=3,
        local_epochs=1,
        non_iid_alpha=0.5,
        samples_per_client=30,
        use_quantization=False,
        aggregation_method='fedavg'
    )
    
    # Run centralized fine-tuning for comparison
    centralized_results = run_centralized_finetune(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_epochs=3,
        batch_size=4,
        lr=2e-4,
        use_quantization=False
    )
    
    # Print comparison
    print_comparison_summary(fed_results, centralized_results)
    
    # Save results
    save_results(fed_results, centralized_results)
    
    # Generate plots
    plot_results(fed_results, centralized_results)
    
    print("\n=== Experiment Complete ===")


if __name__ == "__main__":
    main()
