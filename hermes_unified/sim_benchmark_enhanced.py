"""
Enhanced Simulation Benchmark

Extended simulator with:
1. Dynamic Worker Join/Leave (elastic training)
2. Heterogeneous Networks (asymmetric bandwidth/latency)
3. Sparse Gradient Communication (Top-k)
"""

import math
import random
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import time


class WorkerInfo:
    """Worker information for simulation"""
    def __init__(self, worker_id: int, speed: float = 1.0, join_time: float = 0.0):
        self.worker_id = worker_id
        self.speed = speed
        self.batch_size = 64
        self.compute_time = 0.02 / speed
        self.alive = True
        self.join_time = join_time
        self.leave_time = None


class NetworkLink:
    """Network link information"""
    def __init__(self, src: int, dst: int, bandwidth: float = 100.0, latency: float = 0.0001):
        self.src = src
        self.dst = dst
        self.bandwidth = bandwidth
        self.latency = latency
        self.congestion = 0.0


class EnhancedSimulationConfig:
    """Enhanced simulation configuration"""
    def __init__(self):
        # Basic parameters
        self.num_workers = 4
        self.model_size_mb = 40.0
        self.base_compute_time = 0.02
        self.base_bandwidth = 100.0
        self.base_latency = 0.0001
        self.batch_size_per_worker = 64
        
        # Dynamic worker settings
        self.enable_dynamic_workers = True
        self.worker_join_rate = 0.05
        self.worker_leave_rate = 0.02
        self.min_workers = 2
        self.max_workers = 16
        
        # Heterogeneous network settings
        self.enable_heterogeneous_network = True
        self.topology_type = 'tree'  # 'mesh', 'tree', 'ring'
        self.bw_variation = 0.3
        self.lat_variation = 0.5
        
        # Sparse gradient settings
        self.enable_sparse_gradient = True
        self.topk_ratio = 0.1
        self.enable_error_compensation = True
        
        # Other settings
        self.enable_noise = True
        self.simulation_steps = 200


class EnhancedDistributedSimulator:
    """Enhanced distributed training simulator"""
    
    def __init__(self, config: EnhancedSimulationConfig):
        self.config = config
        self.workers: List[WorkerInfo] = []
        self.network_links: Dict[Tuple[int, int], NetworkLink] = {}
        self.step_count = 0
        self.throughput_history = []
        self.event_log = []
        
        # Error compensation buffer for sparse gradients
        self.error_buffer = np.zeros(int(self.config.model_size_mb * 1024 * 1024 / 4))  # Approx param count
        
        self._initialize_workers()
        self._initialize_network()
    
    def _initialize_workers(self):
        """Initialize workers with heterogeneous speeds"""
        self.workers = []
        for i in range(self.config.num_workers):
            speed = random.uniform(0.5, 1.5) if self.config.enable_dynamic_workers else 1.0
            self.workers.append(WorkerInfo(
                worker_id=i,
                speed=speed,
                join_time=0.0
            ))
    
    def _initialize_network(self):
        """Initialize network topology"""
        self.network_links = {}
        
        if self.config.topology_type == 'mesh':
            self._init_mesh_topology()
        elif self.config.topology_type == 'tree':
            self._init_tree_topology()
        elif self.config.topology_type == 'ring':
            self._init_ring_topology()
        
        if self.config.enable_heterogeneous_network:
            self._add_network_heterogeneity()
    
    def _init_mesh_topology(self):
        """Initialize full mesh topology"""
        num_nodes = self.config.num_workers + 1  # +1 for PS
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    self.network_links[(i, j)] = NetworkLink(
                        src=i,
                        dst=j,
                        bandwidth=self.config.base_bandwidth,
                        latency=self.config.base_latency
                    )
    
    def _init_tree_topology(self):
        """Initialize tree topology (PS as root)"""
        num_nodes = self.config.num_workers + 1
        
        # Root (PS) to leaves (workers)
        for i in range(1, num_nodes):
            self.network_links[(0, i)] = NetworkLink(0, i, bandwidth=150.0, latency=0.00008)
            self.network_links[(i, 0)] = NetworkLink(i, 0, bandwidth=150.0, latency=0.00008)
        
        # Leaves to leaves (lower bandwidth)
        for i in range(1, num_nodes):
            for j in range(1, num_nodes):
                if i != j:
                    self.network_links[(i, j)] = NetworkLink(i, j, bandwidth=60.0, latency=0.00015)
    
    def _init_ring_topology(self):
        """Initialize ring topology"""
        num_nodes = self.config.num_workers
        
        for i in range(num_nodes):
            next_node = (i + 1) % num_nodes
            self.network_links[(i, next_node)] = NetworkLink(i, next_node, bandwidth=100.0, latency=0.0001)
            self.network_links[(next_node, i)] = NetworkLink(next_node, i, bandwidth=100.0, latency=0.0001)
    
    def _add_network_heterogeneity(self):
        """Add heterogeneity to network links"""
        for key, link in self.network_links.items():
            bw_noise = np.random.normal(1.0, self.config.bw_variation)
            link.bandwidth = max(1.0, link.bandwidth * bw_noise)
            
            lat_noise = np.random.normal(1.0, self.config.lat_variation)
            link.latency = max(0.00001, link.latency * lat_noise)
    
    def update_worker_dynamics(self):
        """Update worker join/leave events"""
        if not self.config.enable_dynamic_workers:
            return
        
        # Worker leave
        for worker in self.workers:
            if worker.alive and random.random() < self.config.worker_leave_rate:
                if len(self.get_alive_workers()) > self.config.min_workers:
                    worker.alive = False
                    worker.leave_time = self.step_count
                    self.event_log.append({
                        'step': self.step_count,
                        'type': 'worker_leave',
                        'worker_id': worker.worker_id,
                        'reason': 'random'
                    })
        
        # Worker join
        if random.random() < self.config.worker_join_rate:
            if len(self.get_alive_workers()) < self.config.max_workers:
                new_id = max(w.worker_id for w in self.workers) + 1
                speed = random.uniform(0.5, 1.5)
                self.workers.append(WorkerInfo(
                    worker_id=new_id,
                    speed=speed,
                    join_time=self.step_count
                ))
                self.event_log.append({
                    'step': self.step_count,
                    'type': 'worker_join',
                    'worker_id': new_id,
                    'speed': speed
                })
    
    def get_alive_workers(self) -> List[WorkerInfo]:
        """Get list of alive workers"""
        return [w for w in self.workers if w.alive]
    
    def compute_sparse_gradient_size(self) -> float:
        """Compute effective gradient size with Top-k sparsification"""
        if not self.config.enable_sparse_gradient:
            return self.config.model_size_mb
        
        return self.config.model_size_mb * self.config.topk_ratio
    
    def apply_error_compensation(self, gradient: np.ndarray) -> np.ndarray:
        """Apply error compensation to gradient"""
        if not self.config.enable_error_compensation:
            return gradient
        
        compensated = gradient + self.error_buffer[:gradient.size]
        
        # Update error buffer with discarded values
        if self.config.enable_sparse_gradient:
            n_top = int(gradient.size * self.config.topk_ratio)
            abs_grad = np.abs(compensated)
            threshold = np.sort(abs_grad.flatten())[-n_top]
            mask = abs_grad >= threshold
            self.error_buffer[:gradient.size] = compensated * (1 - mask) * 0.9
        
        return compensated
    
    def compute_communication_time_ps(self) -> float:
        """Compute communication time for PS mode"""
        alive_workers = self.get_alive_workers()
        if not alive_workers:
            return 0.0
        
        msg_size = self.compute_sparse_gradient_size()
        max_upload_time = 0.0
        max_download_time = 0.0
        
        for worker in alive_workers:
            link = self.network_links.get((worker.worker_id + 1, 0))  # +1 because PS is 0, workers are 1+
            if link:
                upload_time = msg_size / link.bandwidth + link.latency
                download_time = self.config.model_size_mb / link.bandwidth + link.latency
                max_upload_time = max(max_upload_time, upload_time)
                max_download_time = max(max_download_time, download_time)
        
        return max_upload_time + max_download_time
    
    def compute_communication_time_ddp(self) -> float:
        """Compute communication time for DDP mode"""
        alive_workers = self.get_alive_workers()
        num_workers = len(alive_workers)
        
        if num_workers <= 1:
            return 0.0
        
        msg_size = self.compute_sparse_gradient_size()
        total_comm_time = 0.0
        
        for i in range(num_workers):
            src = alive_workers[i]
            dst = alive_workers[(i + 1) % num_workers]
            
            link = self.network_links.get((src.worker_id + 1, dst.worker_id + 1))
            if link:
                stage_time = 2 * msg_size / link.bandwidth + link.latency
                total_comm_time += stage_time
        
        return total_comm_time / num_workers * math.log2(num_workers)
    
    def compute_step_time(self, comm_time: float) -> float:
        """Compute step time considering compute-communication overlap"""
        alive_workers = self.get_alive_workers()
        if not alive_workers:
            return float('inf')
        
        compute_time = max(w.compute_time for w in alive_workers)
        return compute_time + comm_time * 0.3  # 30% overlap factor
    
    def compute_throughput(self, step_time: float) -> float:
        """Compute throughput"""
        alive_workers = self.get_alive_workers()
        total_batch = sum(w.batch_size for w in alive_workers)
        return total_batch / step_time if step_time > 0 else 0.0
    
    def run_simulation(self) -> Dict:
        """Run full simulation"""
        results = {
            'throughput_history': [],
            'worker_count_history': [],
            'event_log': [],
            'protocol_comparison': []
        }
        
        for step in range(self.config.simulation_steps):
            self.step_count = step
            
            # Update worker dynamics
            self.update_worker_dynamics()
            
            # Evaluate all protocols
            ps_comm = self.compute_communication_time_ps()
            ddp_comm = self.compute_communication_time_ddp()
            
            ps_step = self.compute_step_time(ps_comm)
            ddp_step = self.compute_step_time(ddp_comm)
            
            ps_throughput = self.compute_throughput(ps_step)
            ddp_throughput = self.compute_throughput(ddp_step)
            
            # Record results
            results['throughput_history'].append({
                'step': step,
                'ps_throughput': ps_throughput,
                'ddp_throughput': ddp_throughput,
                'best_protocol': 'ps' if ps_throughput > ddp_throughput else 'ddp'
            })
            
            results['worker_count_history'].append(len(self.get_alive_workers()))
            
            # Copy events for this step
            step_events = [e for e in self.event_log if e['step'] == step]
            if step_events:
                results['event_log'].extend(step_events)
        
        return results


def generate_plots(results: Dict, output_dir: str = 'results'):
    """Generate visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot throughput over time
        steps = [r['step'] for r in results['throughput_history']]
        ps_throughput = [r['ps_throughput'] for r in results['throughput_history']]
        ddp_throughput = [r['ddp_throughput'] for r in results['throughput_history']]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, ps_throughput, label='PS', color='blue')
        plt.plot(steps, ddp_throughput, label='DDP', color='orange')
        
        # Mark events
        for event in results['event_log']:
            plt.axvline(x=event['step'], color='red' if event['type'] == 'worker_leave' else 'green', 
                       linestyle='--', alpha=0.5)
            label = f"Leave {event['worker_id']}" if event['type'] == 'worker_leave' else f"Join {event['worker_id']}"
            plt.text(event['step'], max(ps_throughput + ddp_throughput) * 0.95, 
                    label, rotation=90, fontsize=8)
        
        plt.xlabel('Step')
        plt.ylabel('Throughput (samples/sec)')
        plt.title('Throughput Over Time with Dynamic Workers')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'throughput_dynamic.png'))
        plt.close()
        
        # Plot worker count over time
        plt.figure(figsize=(12, 4))
        plt.plot(steps, results['worker_count_history'], label='Active Workers', color='green')
        plt.xlabel('Step')
        plt.ylabel('Worker Count')
        plt.title('Worker Count Over Time')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'worker_count.png'))
        plt.close()
        
        print(f"Plots saved to {output_dir}")
        
    except ImportError:
        print("Matplotlib not installed, skipping plots")


def run_sparse_gradient_scan():
    """Run parameter scan for sparse gradient ratios"""
    ratios = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = []
    
    for ratio in ratios:
        config = EnhancedSimulationConfig()
        config.enable_sparse_gradient = True
        config.topk_ratio = ratio
        config.simulation_steps = 100
        
        simulator = EnhancedDistributedSimulator(config)
        sim_results = simulator.run_simulation()
        
        avg_throughput = np.mean([r['ps_throughput'] for r in sim_results['throughput_history']])
        
        results.append({
            'ratio': ratio,
            'avg_throughput': avg_throughput,
            'comm_reduction': 1 - ratio
        })
    
    print("\n=== Sparse Gradient Scan Results ===")
    for r in results:
        print(f"Ratio={r['ratio']:.2f} | Throughput={r['avg_throughput']:.1f} samples/s | Comm Reduction={r['comm_reduction']:.1%}")
    
    return results


if __name__ == "__main__":
    # Run basic simulation
    config = EnhancedSimulationConfig()
    simulator = EnhancedDistributedSimulator(config)
    results = simulator.run_simulation()
    
    # Generate plots
    generate_plots(results)
    
    # Run sparse gradient scan
    sparse_results = run_sparse_gradient_scan()
