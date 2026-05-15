#!/usr/bin/env python3
"""
Hermes Unified Congestion Benchmark

模拟网络拥塞与动态带宽调整策略
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
class CongestionConfig:
    """拥塞模拟配置"""
    num_workers: int = 4
    model_size_mb: float = 40.0
    compute_time_per_worker: float = 0.02
    base_bandwidth_mb_s: float = 100.0
    latency: float = 0.0001
    initial_compression_ratio: float = 0.1
    congestion_threshold: float = 50.0  # MB/s，超过此阈值开始拥塞
    congestion_decay: float = 0.5  # 拥塞时带宽衰减比例
    simulation_steps: int = 50
    congestion_probability: float = 0.2  # 每步发生拥塞的概率


class CongestionSimulator:
    """网络拥塞模拟器"""
    
    def __init__(self, config: CongestionConfig):
        self.config = config
        self.current_bandwidth = config.base_bandwidth_mb_s
        self.current_compression = config.initial_compression_ratio
        self.congested = False
        self.congestion_history = []
        self.compression_history = []
        self.throughput_history = []
        self.step_count = 0
    
    def compute_communication_time(self, compression_ratio: float) -> float:
        """计算通信时间"""
        if self.config.num_workers <= 1:
            return 0.0
        
        msg_size = self.config.model_size_mb * compression_ratio
        comm_time = 2 * msg_size / self.current_bandwidth * math.log2(self.config.num_workers)
        base_latency = self.config.latency * math.log2(self.config.num_workers)
        
        return comm_time + base_latency
    
    def simulate_static_compression(self) -> List[Dict]:
        """模拟静态压缩策略（压缩率固定）"""
        results = []
        
        for step in range(self.config.simulation_steps):
            # 随机发生拥塞
            if random.random() < self.config.congestion_probability:
                self.current_bandwidth = self.config.base_bandwidth_mb_s * self.config.congestion_decay
                self.congested = True
            else:
                self.current_bandwidth = self.config.base_bandwidth_mb_s
                self.congested = False
            
            # 静态压缩：压缩率不变
            compression_ratio = self.config.initial_compression_ratio
            
            comm_time = self.compute_communication_time(compression_ratio)
            step_time = self.config.compute_time_per_worker + comm_time
            throughput = self.config.num_workers * 64 / step_time
            
            results.append({
                'step': step,
                'bandwidth': self.current_bandwidth,
                'compression_ratio': compression_ratio,
                'congested': self.congested,
                'throughput': throughput,
                'step_time': step_time,
                'strategy': 'static'
            })
        
        return results
    
    def simulate_dynamic_compression(self) -> List[Dict]:
        """模拟动态压缩策略（根据拥塞调整压缩率）"""
        results = []
        current_compression = self.config.initial_compression_ratio
        
        for step in range(self.config.simulation_steps):
            # 随机发生拥塞
            if random.random() < self.config.congestion_probability:
                self.current_bandwidth = self.config.base_bandwidth_mb_s * self.config.congestion_decay
                self.congested = True
                # 拥塞时提高压缩率（减少数据量）
                current_compression = max(0.05, current_compression * 0.7)
            else:
                self.current_bandwidth = self.config.base_bandwidth_mb_s
                self.congested = False
                # 非拥塞时逐渐恢复压缩率
                current_compression = min(self.config.initial_compression_ratio, current_compression * 1.1)
            
            comm_time = self.compute_communication_time(current_compression)
            step_time = self.config.compute_time_per_worker + comm_time
            throughput = self.config.num_workers * 64 / step_time
            
            results.append({
                'step': step,
                'bandwidth': self.current_bandwidth,
                'compression_ratio': current_compression,
                'congested': self.congested,
                'throughput': throughput,
                'step_time': step_time,
                'strategy': 'dynamic'
            })
        
        return results
    
    def run_comparison(self) -> Dict:
        """运行静态 vs 动态压缩对比"""
        static_results = self.simulate_static_compression()
        dynamic_results = self.simulate_dynamic_compression()
        
        # 计算统计指标
        static_throughputs = [r['throughput'] for r in static_results]
        dynamic_throughputs = [r['throughput'] for r in dynamic_results]
        
        return {
            'config': {
                'num_workers': self.config.num_workers,
                'model_size_mb': self.config.model_size_mb,
                'base_bandwidth': self.config.base_bandwidth_mb_s,
                'congestion_probability': self.config.congestion_probability,
                'initial_compression': self.config.initial_compression_ratio
            },
            'static_compression': static_results,
            'dynamic_compression': dynamic_results,
            'statistics': {
                'static_avg_throughput': sum(static_throughputs) / len(static_throughputs),
                'static_min_throughput': min(static_throughputs),
                'static_max_throughput': max(static_throughputs),
                'static_std_throughput': (sum((t - sum(static_throughputs)/len(static_throughputs))**2 for t in static_throughputs) / len(static_throughputs))**0.5,
                'dynamic_avg_throughput': sum(dynamic_throughputs) / len(dynamic_throughputs),
                'dynamic_min_throughput': min(dynamic_throughputs),
                'dynamic_max_throughput': max(dynamic_throughputs),
                'dynamic_std_throughput': (sum((t - sum(dynamic_throughputs)/len(dynamic_throughputs))**2 for t in dynamic_throughputs) / len(dynamic_throughputs))**0.5
            }
        }


def print_results(results: Dict):
    """打印结果"""
    print("=" * 80)
    print("Network Congestion and Dynamic Compression Study")
    print("=" * 80)
    print(f"Number of Workers: {results['config']['num_workers']}")
    print(f"Model Size: {results['config']['model_size_mb']} MB")
    print(f"Base Bandwidth: {results['config']['base_bandwidth']} MB/s")
    print(f"Congestion Probability: {results['config']['congestion_probability']*100:.0f}%")
    print(f"Initial Compression: {results['config']['initial_compression']*100:.0f}%")
    print("=" * 80)
    
    # 打印统计数据
    stats = results['statistics']
    print("\n--- Statistics ---")
    print(f"{'Metric':<30} {'Static':<15} {'Dynamic':<15}")
    print("-" * 60)
    print(f"{'Average Throughput':<30} {stats['static_avg_throughput']:<15.2f} {stats['dynamic_avg_throughput']:<15.2f}")
    print(f"{'Minimum Throughput':<30} {stats['static_min_throughput']:<15.2f} {stats['dynamic_min_throughput']:<15.2f}")
    print(f"{'Maximum Throughput':<30} {stats['static_max_throughput']:<15.2f} {stats['dynamic_max_throughput']:<15.2f}")
    print(f"{'Std Deviation':<30} {stats['static_std_throughput']:<15.2f} {stats['dynamic_std_throughput']:<15.2f}")
    
    # 计算改进
    improvement = (stats['dynamic_avg_throughput'] - stats['static_avg_throughput']) / stats['static_avg_throughput'] * 100
    stability_improvement = (stats['static_std_throughput'] - stats['dynamic_std_throughput']) / stats['static_std_throughput'] * 100
    
    print(f"\n--- Improvement ---")
    print(f"Average Throughput Increase: {improvement:.1f}%")
    print(f"Stability Improvement (Lower Std): {stability_improvement:.1f}%")


def plot_results(results: Dict):
    """绘制结果图表"""
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not installed, skipping plot generation")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 吞吐量对比
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
    
    # 压缩率变化
    dynamic_compression = [r['compression_ratio'] for r in results['dynamic_compression']]
    
    ax2.plot(steps, dynamic_compression, label='Compression Ratio', color='red')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Dynamic Compression Ratio Adjustment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.2)
    
    plt.tight_layout()
    plt.savefig('congestion_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to congestion_results.png")


def main():
    parser = argparse.ArgumentParser(description="Congestion Benchmark")
    
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--model-size", type=float, default=40.0, help="Model size (MB)")
    parser.add_argument("--bandwidth", type=float, default=100.0, help="Base bandwidth (MB/s)")
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps")
    parser.add_argument("--congestion-prob", type=float, default=0.2, help="Congestion probability")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # 创建配置
    config = CongestionConfig(
        num_workers=args.workers,
        model_size_mb=args.model_size,
        base_bandwidth_mb_s=args.bandwidth,
        simulation_steps=args.steps,
        congestion_probability=args.congestion_prob
    )
    
    # 运行模拟
    simulator = CongestionSimulator(config)
    results = simulator.run_comparison()
    
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
