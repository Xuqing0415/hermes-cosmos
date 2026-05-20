"""
Visualization Tools for Federated Linear Mode Connectivity

Implements plotting functions for loss landscapes and connectivity evolution.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def plot_interpolation_path(losses: np.ndarray, alphas: Optional[np.ndarray] = None,
                           title: str = 'Interpolation Path Loss',
                           save_path: Optional[str] = None):
    """
    Plot loss along interpolation path.
    
    Args:
        losses: Loss values along interpolation path
        alphas: Alpha values (interpolation weights)
        title: Plot title
        save_path: Path to save plot (None for display)
    """
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, len(losses))
    
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, losses, '-o', color='#1f77b4', markersize=6)
    
    base_loss = max(losses[0], losses[-1])
    plt.axhline(y=base_loss, color='#ff7f0e', linestyle='--', 
                label=f'Base Loss ({base_loss:.4f})')
    
    max_loss_idx = np.argmax(losses)
    plt.scatter(alphas[max_loss_idx], losses[max_loss_idx], 
                color='#d62728', s=100, zorder=5,
                label=f'Max Loss ({losses[max_loss_idx]:.4f})')
    
    plt.xlabel('Interpolation Weight (alpha)')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_connectivity_evolution(round_analyses: List[Dict[str, Any]],
                                title: str = 'Connectivity Evolution',
                                save_path: Optional[str] = None):
    """
    Plot connectivity metrics across rounds.
    
    Args:
        round_analyses: List of round connectivity analyses
        title: Plot title
        save_path: Path to save plot
    """
    rounds = [a['round_idx'] for a in round_analyses]
    mean_connectivity = [a['mean_connectivity'] for a in round_analyses]
    connected_ratios = [a['connected_ratio'] for a in round_analyses]
    mean_spikes = [a['mean_spike'] for a in round_analyses]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(rounds, mean_connectivity, '-o', color='#1f77b4', label='Mean Connectivity')
    ax1.plot(rounds, connected_ratios, '-s', color='#2ca02c', label='Connected Ratio')
    ax1.set_ylabel('Score')
    ax1.set_title('Connectivity Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(rounds, mean_spikes, '-^', color='#ff7f0e', label='Mean Loss Spike')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss Spike')
    ax2.set_title('Loss Spike Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_landscape(losses: np.ndarray, alphas: Optional[np.ndarray] = None,
                        accuracies: Optional[np.ndarray] = None,
                        title: str = 'Loss Landscape',
                        save_path: Optional[str] = None):
    """
    Plot detailed loss landscape with optional accuracy.
    
    Args:
        losses: Loss values
        alphas: Alpha values
        accuracies: Accuracy values (optional)
        title: Plot title
        save_path: Path to save plot
    """
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, len(losses))
    
    if accuracies is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax1.plot(alphas, losses, '-o', color='#1f77b4')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(alphas, accuracies, '-s', color='#2ca02c')
        ax2.set_xlabel('Interpolation Weight (alpha)')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(alphas, losses, '-o', color='#1f77b4')
        plt.xlabel('Interpolation Weight (alpha)')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_client_connectivity_distribution(connectivity_scores: List[float],
                                         round_idx: int,
                                         save_path: Optional[str] = None):
    """
    Plot distribution of client connectivity scores.
    
    Args:
        connectivity_scores: List of connectivity scores
        round_idx: Round index
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 5))
    
    n, bins, patches = plt.hist(connectivity_scores, bins=10, 
                               color='#1f77b4', alpha=0.7)
    
    plt.axvline(x=0.5, color='#ff7f0e', linestyle='--', 
                label='Connectivity Threshold (0.5)')
    
    plt.xlabel('Connectivity Score')
    plt.ylabel('Number of Clients')
    plt.title(f'Client Connectivity Distribution - Round {round_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_multiple_interpolation_paths(paths: List[Dict[str, Any]],
                                     title: str = 'Multiple Interpolation Paths',
                                     save_path: Optional[str] = None):
    """
    Plot multiple interpolation paths together.
    
    Args:
        paths: List of dictionaries with 'losses' and optional 'label'
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
    
    for i, path in enumerate(paths):
        losses = path['losses']
        alphas = path.get('alphas', np.linspace(0.0, 1.0, len(losses)))
        label = path.get('label', f'Path {i+1}')
        
        plt.plot(alphas, losses, '-o', color=colors[i], label=label, markersize=4)
    
    plt.xlabel('Interpolation Weight (alpha)')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_heatmap_connectivity(client_ids: List[int], rounds: List[int],
                              connectivity_matrix: np.ndarray,
                              save_path: Optional[str] = None):
    """
    Plot heatmap of client connectivity across rounds.
    
    Args:
        client_ids: List of client IDs
        rounds: List of rounds
        connectivity_matrix: 2D array [rounds x clients] of connectivity scores
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    im = plt.imshow(connectivity_matrix.T, cmap='viridis', aspect='auto')
    
    plt.xticks(np.arange(len(rounds)), rounds)
    plt.yticks(np.arange(len(client_ids)), client_ids)
    
    plt.colorbar(im, label='Connectivity Score')
    
    plt.xlabel('Round')
    plt.ylabel('Client ID')
    plt.title('Client Connectivity Heatmap')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_report(results: Dict[str, Any], output_dir: str = '.'):
    """
    Generate a comprehensive report with all visualizations.
    
    Args:
        results: LMC analysis results
        output_dir: Directory to save report
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    if 'round_results' in results:
        plot_connectivity_evolution(
            results['round_results'],
            save_path=os.path.join(output_dir, 'connectivity_evolution.png')
        )
        
        for round_analysis in results['round_results']:
            round_idx = round_analysis['round_idx']
            scores = [r['connectivity_score'] for r in round_analysis.get('client_results', [])]
            
            if scores:
                plot_client_connectivity_distribution(
                    scores, round_idx,
                    save_path=os.path.join(output_dir, f'client_dist_round_{round_idx}.png')
                )
    
    if 'cross_round_results' in results:
        for cross in results['cross_round_results']:
            analysis = cross['analysis']
            round1, round2 = cross['round1'], cross['round2']
            
            plot_interpolation_path(
                analysis['losses'], analysis['alphas'],
                title=f'Cross-Round Interpolation: Round {round1} -> {round2}',
                save_path=os.path.join(output_dir, f'cross_round_{round1}_{round2}.png')
            )
    
    print(f"Report generated in {output_dir}")