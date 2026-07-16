#!/usr/bin/env python3
"""
Hermes Unified Overlap Benchmark


"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class OverlapConfig:
    """"""
    num_workers: int = 4
    model_size_mb: float = 40.0
    compute_time_per_worker: float = 0.02  #  worker 
    bandwidth_mb_s: float = 100.0
    latency: float = 0.0001
    compression_ratio: float = 1.0
    num_layers: int = 10  # 
    overlap_factors: List[float] = None  # 
    
    def __post_init__(self):
        if self.overlap_factors is None:
            self.overlap_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class OverlapSimulator:
    """"""
    
    def __init__(self, config: OverlapConfig):
        self.config = config
    
    def compute_communication_time(self) -> float:
        """DDP Ring All-Reduce"""
        if self.config.num_workers <= 1:
            return 0.0
        
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        comm_time = 2 * msg_size / self.config.bandwidth_mb_s * math.log2(self.config.num_workers)
        base_latency = self.config.latency * math.log2(self.config.num_workers)
        
        return comm_time + base_latency
    
    def simulate_serial_execution(self) -> Dict:
        """=0"""
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 
        step_time = compute_time + comm_time
        throughput = self.config.num_workers * 64 / step_time
        
        return {
            'overlap_factor': 0.0,
            'step_time': step_time,
            'throughput': throughput,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'description': ''
        }
    
    def simulate_ideal_overlap(self) -> Dict:
        """=1"""
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 
        step_time = max(compute_time, comm_time)
        throughput = self.config.num_workers * 64 / step_time
        
        return {
            'overlap_factor': 1.0,
            'step_time': step_time,
            'throughput': throughput,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'description': ''
        }
    
    def simulate_partial_overlap(self, overlap_factor: float) -> Dict:
        """
        
        
        Args:
            overlap_factor: 0~1
        
        Returns:
            results: 
        """
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 
        num_segments = max(2, int(self.config.num_layers / 2))
        segment_compute_time = compute_time / num_segments
        
        # 
        # 
        total_time = segment_compute_time
        
        # 
        for _ in range(num_segments - 1):
            #  =  / 
            segment_comm_time = (comm_time / num_segments) * (1 - overlap_factor)
            total_time += max(segment_compute_time, segment_comm_time)
        
        # 
        remaining_comm_time = comm_time * overlap_factor
        total_time += remaining_comm_time
        
        throughput = self.config.num_workers * 64 / total_time
        
        return {
            'overlap_factor': overlap_factor,
            'step_time': total_time,
            'throughput': throughput,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'num_segments': num_segments,
            'description': f' ({overlap_factor*100:.0f}%)'
        }
    
    def simulate_dynamic_overlap(self) -> Dict:
        """
        
        
        
        -  << 
        -  >> 
        - 
        """
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # /
        comm_ratio = comm_time / (compute_time + comm_time)
        
        # 
        # 
        dynamic_overlap = min(1.0, max(0.0, 2 * comm_ratio - 0.3))
        
        return self.simulate_partial_overlap(dynamic_overlap)
    
    def run_overlap_study(self) -> Dict:
        """"""
        results = {
            'config': {
                'num_workers': self.config.num_workers,
                'model_size_mb': self.config.model_size_mb,
                'bandwidth_mb_s': self.config.bandwidth_mb_s,
                'compute_time': self.config.compute_time_per_worker
            },
            'fixed_overlap': [],
            'dynamic_overlap': None
        }
        
        # 
        for factor in self.config.overlap_factors:
            result = self.simulate_partial_overlap(factor)
            results['fixed_overlap'].append(result)
        
        # 
        results['dynamic_overlap'] = self.simulate_dynamic_overlap()
        
        return results


def print_results(results: Dict):
    """"""
    print("=" * 80)
    print("Communication-Compute Overlap Study")
    print("=" * 80)
    print(f"Number of Workers: {results['config']['num_workers']}")
    print(f"Model Size: {results['config']['model_size_mb']} MB")
    print(f"Bandwidth: {results['config']['bandwidth_mb_s']} MB/s")
    print(f"Compute Time: {results['config']['compute_time']:.4f} s")
    print("=" * 80)
    
    # 
    print("\n--- Fixed Overlap Factors ---")
    print(f"{'Overlap':<10} {'Step Time':<15} {'Throughput':<15} {'Description'}")
    print("-" * 60)
    for result in results['fixed_overlap']:
        print(f"{result['overlap_factor']:<10.1f} {result['step_time']:<15.4f} "
              f"{result['throughput']:<15.2f} {result['description']}")
    
    # 
    print("\n--- Dynamic Overlap ---")
    dynamic = results['dynamic_overlap']
    print(f"Optimal Overlap Factor: {dynamic['overlap_factor']:.3f}")
    print(f"Step Time: {dynamic['step_time']:.4f} s")
    print(f"Throughput: {dynamic['throughput']:.2f} samples/sec")
    print(f"Description: {dynamic['description']}")
    
    # 
    best_result = max(results['fixed_overlap'], key=lambda x: x['throughput'])
    print(f"\n--- Best Fixed Overlap ---")
    print(f"Best Overlap Factor: {best_result['overlap_factor']:.1f}")
    print(f"Max Throughput: {best_result['throughput']:.2f} samples/sec")
    print(f"Improvement over Serial: {(best_result['throughput'] - results['fixed_overlap'][0]['throughput']) / results['fixed_overlap'][0]['throughput'] * 100:.1f}%")


def plot_results(results: Dict):
    """"""
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not installed, skipping plot generation")
        return
    
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
    
    # 
    best_result = max(results['fixed_overlap'], key=lambda x: x['throughput'])
    ax.scatter(best_result['overlap_factor'], best_result['throughput'],
               color='green', s=100, label='Best Fixed', zorder=5)
    
    ax.set_xlabel('Overlap Factor')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('Throughput vs Overlap Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('overlap_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to overlap_results.png")


def main():
    parser = argparse.ArgumentParser(description="Overlap Benchmark")
    
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--model-size", type=float, default=40.0, help="Model size (MB)")
    parser.add_argument("--bandwidth", type=float, default=100.0, help="Bandwidth (MB/s)")
    parser.add_argument("--compute-time", type=float, default=0.02, help="Compute time per worker (s)")
    parser.add_argument("--layers", type=int, default=10, help="Number of model layers")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # 
    config = OverlapConfig(
        num_workers=args.workers,
        model_size_mb=args.model_size,
        bandwidth_mb_s=args.bandwidth,
        compute_time_per_worker=args.compute_time,
        num_layers=args.layers
    )
    
    # 
    simulator = OverlapSimulator(config)
    results = simulator.run_overlap_study()
    
    # 
    print_results(results)
    plot_results(results)
    
    # 
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
