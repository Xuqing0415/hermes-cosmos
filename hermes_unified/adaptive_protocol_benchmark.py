#!/usr/bin/env python3
"""
Hermes Unified Adaptive Protocol Benchmark

模拟自适应通信协议切换策略（PS ↔ Gossip ↔ DDP）
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
class ProtocolConfig:
    """自适应协议配置"""
    initial_workers: int = 4
    max_workers: int = 32
    min_workers: int = 4
    model_size_mb: float = 40.0
    compute_time_per_worker: float = 0.02
    bandwidth_mb_s: float = 100.0
    latency: float = 0.0001
    compression_ratio: float = 1.0
    simulation_steps: int = 100
    switch_interval: int = 10  # 每多少步评估一次并可能切换
    worker_change_interval: int = 20  # 每多少步改变worker数量


class AdaptiveProtocolSimulator:
    """自适应通信协议模拟器"""
    
    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.current_workers = config.initial_workers
        self.current_protocol = 'ddp'  # 初始协议
        self.protocol_history = []
        self.worker_history = []
        self.throughput_history = []
        self.step_count = 0
    
    def compute_ps_comm_time(self) -> float:
        """计算PS模式通信时间"""
        msg_size_send = self.config.model_size_mb * self.config.compression_ratio
        msg_size_recv = self.config.model_size_mb
        
        upload_time = msg_size_send / self.config.bandwidth_mb_s
        download_time = msg_size_recv / self.config.bandwidth_mb_s
        base_latency = self.config.latency * 2
        
        return upload_time + download_time + base_latency
    
    def compute_ddp_comm_time(self) -> float:
        """计算DDP模式通信时间"""
        if self.current_workers <= 1:
            return 0.0
        
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        comm_time = 2 * msg_size / self.config.bandwidth_mb_s * math.log2(self.current_workers)
        base_latency = self.config.latency * math.log2(self.current_workers)
        
        return comm_time + base_latency
    
    def compute_gossip_comm_time(self) -> float:
        """计算Gossip模式通信时间"""
        k = max(2, int(math.log2(self.current_workers)))
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        
        comm_time = k * (2 * msg_size / self.config.bandwidth_mb_s)
        base_latency = self.config.latency * k
        
        return comm_time + base_latency
    
    def evaluate_protocol(self, protocol: str) -> float:
        """评估指定协议的吞吐量"""
        compute_time = self.config.compute_time_per_worker
        
        if protocol == 'ps':
            comm_time = self.compute_ps_comm_time()
        elif protocol == 'ddp':
            comm_time = self.compute_ddp_comm_time()
        elif protocol == 'gossip':
            comm_time = self.compute_gossip_comm_time()
        else:
            comm_time = self.compute_ddp_comm_time()
        
        step_time = compute_time + comm_time
        throughput = self.current_workers * 64 / step_time
        
        return throughput
    
    def select_best_protocol(self) -> str:
        """选择当前条件下最优的协议"""
        protocols = ['ps', 'ddp', 'gossip']
        throughputs = {p: self.evaluate_protocol(p) for p in protocols}
        
        # 选择吞吐量最高的协议
        best_protocol = max(throughputs, key=throughputs.get)
        best_throughput = throughputs[best_protocol]
        
        return best_protocol, best_throughput, throughputs
    
    def simulate_adaptive_switching(self) -> List[Dict]:
        """模拟自适应协议切换"""
        results = []
        protocol_counts = {'ps': 0, 'ddp': 0, 'gossip': 0}
        
        for step in range(self.config.simulation_steps):
            # 周期性改变worker数量
            if step > 0 and step % self.config.worker_change_interval == 0:
                # 模拟worker数量变化
                if self.current_workers >= self.config.max_workers:
                    # 减少worker
                    self.current_workers = max(self.config.min_workers, self.current_workers - 4)
                else:
                    # 增加worker
                    self.current_workers = min(self.config.max_workers, self.current_workers + 4)
            
            # 周期性评估并可能切换协议
            if step % self.config.switch_interval == 0:
                best_protocol, best_throughput, all_throughputs = self.select_best_protocol()
                
                if best_protocol != self.current_protocol:
                    print(f"[Step {step}] Switching from {self.current_protocol.upper()} to {best_protocol.upper()}")
                    print(f"  Throughputs: PS={all_throughputs['ps']:.1f}, DDP={all_throughputs['ddp']:.1f}, Gossip={all_throughputs['gossip']:.1f}")
                
                self.current_protocol = best_protocol
            
            # 计算当前协议的吞吐量
            current_throughput = self.evaluate_protocol(self.current_protocol)
            
            protocol_counts[self.current_protocol] += 1
            
            results.append({
                'step': step,
                'num_workers': self.current_workers,
                'protocol': self.current_protocol,
                'throughput': current_throughput,
                'switch_made': step % self.config.switch_interval == 0
            })
        
        return results, protocol_counts
    
    def run_simulation(self) -> Dict:
        """运行完整模拟"""
        results, protocol_counts = self.simulate_adaptive_switching()
        
        # 计算统计数据
        throughputs = [r['throughput'] for r in results]
        workers_list = [r['num_workers'] for r in results]
        
        return {
            'config': {
                'initial_workers': self.config.initial_workers,
                'max_workers': self.config.max_workers,
                'min_workers': self.config.min_workers,
                'model_size_mb': self.config.model_size_mb,
                'bandwidth': self.config.bandwidth_mb_s
            },
            'results': results,
            'protocol_counts': protocol_counts,
            'statistics': {
                'avg_throughput': sum(throughputs) / len(throughputs),
                'min_throughput': min(throughputs),
                'max_throughput': max(throughputs),
                'avg_workers': sum(workers_list) / len(workers_list)
            }
        }


def print_results(results: Dict):
    """打印结果"""
    print("=" * 80)
    print("Adaptive Protocol Switching Simulation")
    print("=" * 80)
    print(f"Initial Workers: {results['config']['initial_workers']}")
    print(f"Max Workers: {results['config']['max_workers']}")
    print(f"Min Workers: {results['config']['min_workers']}")
    print(f"Model Size: {results['config']['model_size_mb']} MB")
    print(f"Bandwidth: {results['config']['bandwidth']} MB/s")
    print("=" * 80)
    
    # 打印统计数据
    stats = results['statistics']
    print("\n--- Statistics ---")
    print(f"Average Throughput: {stats['avg_throughput']:.2f} samples/sec")
    print(f"Minimum Throughput: {stats['min_throughput']:.2f} samples/sec")
    print(f"Maximum Throughput: {stats['max_throughput']:.2f} samples/sec")
    print(f"Average Workers: {stats['avg_workers']:.1f}")
    
    # 打印协议使用统计
    counts = results['protocol_counts']
    total = sum(counts.values())
    print("\n--- Protocol Usage ---")
    print(f"PS: {counts['ps']} steps ({counts['ps']/total*100:.1f}%)")
    print(f"DDP: {counts['ddp']} steps ({counts['ddp']/total*100:.1f}%)")
    print(f"Gossip: {counts['gossip']} steps ({counts['gossip']/total*100:.1f}%)")
    
    # 打印切换日志摘要
    print("\n--- Switch Log Summary ---")
    current_protocol = None
    switches = []
    
    for r in results['results']:
        if current_protocol is None:
            current_protocol = r['protocol']
            start_step = r['step']
        elif r['protocol'] != current_protocol:
            switches.append(f"  Step {start_step}-{r['step']-1}: {current_protocol.upper()} ({r['step']-start_step} steps)")
            current_protocol = r['protocol']
            start_step = r['step']
    
    if current_protocol:
        switches.append(f"  Step {start_step}-{results['results'][-1]['step']}: {current_protocol.upper()} ({results['results'][-1]['step']-start_step+1} steps)")
    
    print(f"Total switches: {len(switches)-1}")
    for s in switches:
        print(s)


def plot_results(results: Dict):
    """绘制结果图表"""
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not installed, skipping plot generation")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 吞吐量和worker数量
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
    
    # 协议切换
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
    plt.savefig('adaptive_protocol_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to adaptive_protocol_results.png")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Protocol Benchmark")
    
    parser.add_argument("--initial-workers", type=int, default=4, help="Initial number of workers")
    parser.add_argument("--max-workers", type=int, default=32, help="Maximum workers")
    parser.add_argument("--min-workers", type=int, default=4, help="Minimum workers")
    parser.add_argument("--model-size", type=float, default=40.0, help="Model size (MB)")
    parser.add_argument("--bandwidth", type=float, default=100.0, help="Bandwidth (MB/s)")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # 创建配置
    config = ProtocolConfig(
        initial_workers=args.initial_workers,
        max_workers=args.max_workers,
        min_workers=args.min_workers,
        model_size_mb=args.model_size,
        bandwidth_mb_s=args.bandwidth,
        simulation_steps=args.steps
    )
    
    # 运行模拟
    simulator = AdaptiveProtocolSimulator(config)
    results = simulator.run_simulation()
    
    # 输出结果
    print_results(results)
    plot_results(results)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
