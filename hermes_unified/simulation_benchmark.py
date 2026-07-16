#!/usr/bin/env python3
"""
Hermes Unified Simulation Benchmark

 Python  GPU 


1.  PSDDPAll-ReduceGossip 
2. 
3. 
4. 
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class SimulationConfig:
    """"""
    num_workers_list: List[int] = None
    model_size_mb: float = 40.0  # MB10M * 4 bytes = 40MB
    compute_time_per_worker: float = 0.02  # worker
    bandwidth_mb_s: float = 100.0  # MB/s
    latency: float = 0.0001  # 
    gossip_neighbors: int = 2  # Gossip
    compression_ratio: float = 1.0  # 
    overlap_factor: float = 0.0  # 0=1=
    batch_size_per_worker: int = 64  # workerbatch size
    enable_noise: bool = False  # 


class DistributedSimulator:
    """"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def compute_communication_time_ps(self, num_workers: int) -> float:
        """
        
        
        PSworkerPSworker
        """
        # 
        msg_size_send = self.config.model_size_mb * self.config.compression_ratio
        # 
        msg_size_recv = self.config.model_size_mb
        
        # worker
        upload_time = msg_size_send / self.config.bandwidth_mb_s
        # PSworker
        download_time = msg_size_recv / self.config.bandwidth_mb_s
        
        # 
        base_latency = self.config.latency * 2  # +
        
        return upload_time + download_time + base_latency
    
    def compute_communication_time_ddp(self, num_workers: int) -> float:
        """
        DDPRing All-Reduce
        
        Ring All-Reduce = 2 * (num_workers - 1) / num_workers * model_size
        comm_time = 2 * model_size / bandwidth * log2(num_workers)
        """
        if num_workers <= 1:
            return 0.0
        
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        
        # Ring All-Reduce 
        #  (num_workers-1)  (num_workers-1) 
        #  = 2 * (num_workers-1) * msg_size / (num_workers * bandwidth)
        # 2 * msg_size / bandwidth * log2(num_workers)
        
        comm_time = 2 * msg_size / self.config.bandwidth_mb_s * math.log2(num_workers)
        
        # 
        base_latency = self.config.latency * math.log2(num_workers)
        
        return comm_time + base_latency
    
    def compute_communication_time_gossip(self, num_workers: int) -> float:
        """
        Gossip
        
        Gossipworkerk
        """
        k = self.config.gossip_neighbors
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        
        # model_sizemodel_size
        # worker
        comm_time = k * (2 * msg_size / self.config.bandwidth_mb_s)
        
        # 
        base_latency = self.config.latency * k
        
        return comm_time + base_latency
    
    def compute_step_time(self, num_workers: int, comm_time: float) -> float:
        """
        step
        
        
        step_time = max(compute_time, comm_time * overlap) + (1 - overlap) * (compute_time + comm_time)
        """
        # workerworker
        compute_time = self.config.compute_time_per_worker
        
        # 
        overlap = self.config.overlap_factor
        step_time = max(compute_time, comm_time * overlap) + (1 - overlap) * (compute_time + comm_time)
        
        # 
        if self.config.enable_noise:
            noise = random.uniform(-0.05, 0.05)
            step_time *= (1 + noise)
        
        return max(step_time, 0.0001)  # 
    
    def compute_throughput(self, num_workers: int, step_time: float) -> float:
        """
        samples/sec
        
        throughput = batch_size * num_workers / step_time
        """
        return self.config.batch_size_per_worker * num_workers / step_time
    
    def run_simulation(self) -> Dict[str, List[Dict]]:
        """
        
        
        Returns:
            results: 
        """
        results = {
            'ps': [],
            'ddp': [],
            'gossip': []
        }
        
        for num_workers in self.config.num_workers_list:
            # 
            ps_comm_time = self.compute_communication_time_ps(num_workers)
            ps_step_time = self.compute_step_time(num_workers, ps_comm_time)
            ps_throughput = self.compute_throughput(num_workers, ps_step_time)
            ps_comm_ratio = (ps_comm_time / ps_step_time) * 100
            
            results['ps'].append({
                'num_workers': num_workers,
                'comm_time': ps_comm_time,
                'step_time': ps_step_time,
                'throughput': ps_throughput,
                'comm_ratio': ps_comm_ratio
            })
            
            # DDP
            ddp_comm_time = self.compute_communication_time_ddp(num_workers)
            ddp_step_time = self.compute_step_time(num_workers, ddp_comm_time)
            ddp_throughput = self.compute_throughput(num_workers, ddp_step_time)
            ddp_comm_ratio = (ddp_comm_time / ddp_step_time) * 100
            
            results['ddp'].append({
                'num_workers': num_workers,
                'comm_time': ddp_comm_time,
                'step_time': ddp_step_time,
                'throughput': ddp_throughput,
                'comm_ratio': ddp_comm_ratio
            })
            
            # Gossip
            gossip_comm_time = self.compute_communication_time_gossip(num_workers)
            gossip_step_time = self.compute_step_time(num_workers, gossip_comm_time)
            gossip_throughput = self.compute_throughput(num_workers, gossip_step_time)
            gossip_comm_ratio = (gossip_comm_time / gossip_step_time) * 100
            
            results['gossip'].append({
                'num_workers': num_workers,
                'comm_time': gossip_comm_time,
                'step_time': gossip_step_time,
                'throughput': gossip_throughput,
                'comm_ratio': gossip_comm_ratio
            })
        
        return results


def print_results(results: Dict[str, List[Dict]], config: SimulationConfig):
    """"""
    print("=" * 80)
    print(f"Hermes Unified Simulation Benchmark Results")
    print("=" * 80)
    print(f"Model Size: {config.model_size_mb} MB")
    print(f"Bandwidth: {config.bandwidth_mb_s} MB/s")
    print(f"Compute Time per Worker: {config.compute_time_per_worker:.4f} s")
    print(f"Compression Ratio: {config.compression_ratio * 100:.0f}%")
    print(f"Overlap Factor: {config.overlap_factor}")
    print("=" * 80)
    
    # Print results for each mode
    for mode, data in results.items():
        print(f"\n--- {mode.upper()} Mode ---")
        print(f"{'Workers':<10} {'Throughput':<15} {'Step Time':<15} {'Comm Ratio':<12}")
        print("-" * 52)
        for entry in data:
            print(f"{entry['num_workers']:<10} {entry['throughput']:<15.2f} "
                  f"{entry['step_time']:<15.4f} {entry['comm_ratio']:<12.1f}%")


def plot_results(results: Dict[str, List[Dict]], config: SimulationConfig):
    """"""
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not installed, skipping plot generation")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 
    colors = {'ps': 'blue', 'ddp': 'green', 'gossip': 'purple'}
    labels = {'ps': 'PS ()', 'ddp': 'DDP (All-Reduce)', 'gossip': 'Gossip'}
    
    for mode, data in results.items():
        workers = [e['num_workers'] for e in data]
        throughput = [e['throughput'] for e in data]
        ax1.plot(workers, throughput, marker='o', label=labels[mode], color=colors[mode])
    
    ax1.set_xlabel('Worker')
    ax1.set_ylabel(' (samples/sec)')
    ax1.set_title('')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 
    for mode, data in results.items():
        workers = [e['num_workers'] for e in data]
        comm_ratio = [e['comm_ratio'] for e in data]
        ax2.plot(workers, comm_ratio, marker='s', label=labels[mode], color=colors[mode])
    
    ax2.set_xlabel('Worker')
    ax2.set_ylabel(' (%)')
    ax2.set_title('Worker')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
    print("\n  simulation_results.png")


def run_simulation_with_config(config: SimulationConfig) -> Dict[str, List[Dict]]:
    """"""
    simulator = DistributedSimulator(config)
    results = simulator.run_simulation()
    
    print_results(results, config)
    plot_results(results, config)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Hermes Unified Simulation Benchmark")
    
    # 
    parser.add_argument("--model-size", type=float, default=40.0, help="Model size (MB)")
    parser.add_argument("--compute-time", type=float, default=0.02, help="Compute time per worker (seconds)")
    
    # Network parameters
    parser.add_argument("--bandwidth", type=float, default=100.0, help="Network bandwidth (MB/s)")
    parser.add_argument("--latency", type=float, default=0.0001, help="Network latency (seconds)")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per worker")
    parser.add_argument("--workers", type=str, default="1,2,4,8,16,32", 
                        help="List of worker counts (comma separated)")
    
    # Optimization parameters
    parser.add_argument("--compression", type=float, default=1.0, help="Gradient compression ratio (0.1 equals 10 percent)")
    parser.add_argument("--overlap", type=float, default=0.0, help="Compute-comm overlap factor (0-1)")
    
    # Gossip parameters
    parser.add_argument("--gossip-neighbors", type=int, default=2, help="Number of gossip neighbors")
    
    # Other
    parser.add_argument("--noise", action="store_true", help="Add random noise")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # worker
    workers_list = [int(w.strip()) for w in args.workers.split(',')]
    
    # 
    config = SimulationConfig(
        num_workers_list=workers_list,
        model_size_mb=args.model_size,
        compute_time_per_worker=args.compute_time,
        bandwidth_mb_s=args.bandwidth,
        latency=args.latency,
        gossip_neighbors=args.gossip_neighbors,
        compression_ratio=args.compression,
        overlap_factor=args.overlap,
        batch_size_per_worker=args.batch_size,
        enable_noise=args.noise
    )
    
    # 
    results = run_simulation_with_config(config)
    
    # 
    if args.output:
        output_data = {
            'config': {
                'model_size_mb': config.model_size_mb,
                'compute_time_per_worker': config.compute_time_per_worker,
                'bandwidth_mb_s': config.bandwidth_mb_s,
                'latency': config.latency,
                'compression_ratio': config.compression_ratio,
                'overlap_factor': config.overlap_factor,
                'batch_size_per_worker': config.batch_size_per_worker
            },
            'results': results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  {args.output}")


if __name__ == "__main__":
    main()
