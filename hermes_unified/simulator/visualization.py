#!/usr/bin/env python3
"""
Visualization Module


"""

import os
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_heterogeneous_results(results: Dict, output_dir: str = "results") -> None:
    """"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 
    strategies = ['Static Sharding', 'Dynamic Load Balancing']
    throughputs = [
        results['static_sharding']['throughput'],
        results['dynamic_load_balancing']['throughput']
    ]
    
    ax1.bar(strategies, throughputs, color=['blue', 'green'])
    ax1.set_ylabel('Throughput (samples/sec)')
    ax1.set_title('Throughput Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Worker 
    worker_ids = [f'Worker {i+1}' for i in range(results['config']['num_workers'])]
    static_times = results['static_sharding']['worker_times']
    dynamic_times = results['dynamic_load_balancing']['worker_times']
    
    x = range(len(worker_ids))
    width = 0.35
    
    ax2.bar([i - width/2 for i in x], static_times, width, label='Static', color='blue')
    ax2.bar([i + width/2 for i in x], dynamic_times, width, label='Dynamic', color='green')
    ax2.set_xlabel('Workers')
    ax2.set_ylabel('Compute Time (s)')
    ax2.set_title('Worker Compute Time Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(worker_ids)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heterogeneous_results.png'), dpi=150)
    plt.close()


def plot_overlap_results(results: Dict, output_dir: str = "results") -> None:
    """"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 
    factors = [r['overlap_factor'] for r in results['fixed_overlap']]
    throughputs = [r['throughput'] for r in results['fixed_overlap']]
    
    # 
    ax.plot(factors, throughputs, marker='o', label='Fixed Overlap', color='blue')
    
    # 
    dynamic = results['dynamic_overlap']
    ax.scatter(dynamic['overlap_factor'], dynamic['throughput'], 
               color='red', s=100, label='Dynamic Overlap', zorder=5)
    
    ax.set_xlabel('Overlap Factor')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('Throughput vs Overlap Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overlap_results.png'), dpi=150)
    plt.close()


def plot_congestion_results(results: Dict, output_dir: str = "results") -> None:
    """"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 
    steps = [r['step'] for r in results['static_compression']]
    static_throughput = [r['throughput'] for r in results['static_compression']]
    dynamic_throughput = [r['throughput'] for r in results['dynamic_compression']]
    
    ax1.plot(steps, static_throughput, label='Static Compression', color='blue')
    ax1.plot(steps, dynamic_throughput, label='Dynamic Compression', color='green')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Throughput (samples/sec)')
    ax1.set_title('Throughput Comparison During Congestion')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 
    dynamic_compression = [r['compression_ratio'] for r in results['dynamic_compression']]
    
    ax2.plot(steps, dynamic_compression, label='Compression Ratio', color='red')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Dynamic Compression Ratio Adjustment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'congestion_results.png'), dpi=150)
    plt.close()


def plot_protocol_switch_results(results: Dict, output_dir: str = "results") -> None:
    """"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # worker
    steps = [r['step'] for r in results['results']]
    throughputs = [r['throughput'] for r in results['results']]
    workers = [r['num_workers'] for r in results['results']]
    
    ax1.plot(steps, throughputs, label='Throughput', color='blue')
    ax1.set_ylabel('Throughput (samples/sec)')
    ax1.set_title('Throughput vs Step')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, workers, label='Workers', color='green')
    ax1_twin.set_ylabel('Number of Workers')
    ax1_twin.legend(loc='upper right')
    
    # 
    protocol_colors = {'ps': 'red', 'ddp': 'green', 'gossip': 'purple'}
    protocol_y = {'ps': 0, 'ddp': 1, 'gossip': 2}
    
    for r in results['results']:
        ax2.scatter(r['step'], protocol_y[r['protocol']], 
                    color=protocol_colors[r['protocol']], s=50)
    
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['PS', 'DDP', 'Gossip'])
    ax2.set_xlabel('Step')
    ax2.set_title('Protocol Switching Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'protocol_switch_results.png'), dpi=150)
    plt.close()


def plot_results(results: Dict, analysis_type: str, output_dir: str = "results") -> None:
    """"""
    os.makedirs(output_dir, exist_ok=True)
    
    if analysis_type == 'heterogeneous':
        plot_heterogeneous_results(results, output_dir)
    elif analysis_type == 'overlap':
        plot_overlap_results(results, output_dir)
    elif analysis_type == 'congestion':
        plot_congestion_results(results, output_dir)
    elif analysis_type == 'protocol_switch':
        plot_protocol_switch_results(results, output_dir)
    else:
        print(f"Unknown analysis type: {analysis_type}")
