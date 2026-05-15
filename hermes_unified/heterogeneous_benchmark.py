#!/usr/bin/env python3
"""
Hermes Unified Heterogeneous Benchmark

模拟异构 worker 环境下的负载均衡调度策略
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
class WorkerConfig:
    """单个 worker 的配置"""
    id: int
    compute_speed: float  # 计算速度相对于基准的倍数（0.5x, 1x, 2x等）
    batch_size: int = 64  # 当前分配的 batch size


@dataclass
class HeterogeneousConfig:
    """异构模拟配置"""
    num_workers: int = 4
    model_size_mb: float = 40.0
    base_compute_time: float = 0.02  # 基准计算时间（1x速度的worker）
    bandwidth_mb_s: float = 100.0
    latency: float = 0.0001
    compression_ratio: float = 1.0
    
    # 异构配置
    speed_distribution: str = "random"  # random, uniform, tiered
    min_speed: float = 0.5
    max_speed: float = 2.0


class HeterogeneousSimulator:
    """异构分布式训练模拟器"""
    
    def __init__(self, config: HeterogeneousConfig):
        self.config = config
        self.workers = self._initialize_workers()
        self.step_count = 0
    
    def _initialize_workers(self) -> List[WorkerConfig]:
        """初始化异构 workers"""
        workers = []
        
        for i in range(self.config.num_workers):
            if self.config.speed_distribution == "uniform":
                speed = 1.0
            elif self.config.speed_distribution == "tiered":
                # 分层分布：一半快，一半慢
                if i < self.config.num_workers // 2:
                    speed = self.config.min_speed
                else:
                    speed = self.config.max_speed
            else:  # random
                speed = random.uniform(self.config.min_speed, self.config.max_speed)
            
            workers.append(WorkerConfig(
                id=i,
                compute_speed=speed,
                batch_size=64  # 默认 batch size
            ))
        
        return workers
    
    def compute_worker_time(self, worker: WorkerConfig, batch_size: int) -> float:
        """计算单个 worker 处理指定 batch 的时间"""
        # 计算时间与 batch size 成正比，与计算速度成反比
        base_time = self.config.base_compute_time * (batch_size / 64)
        return base_time / worker.compute_speed
    
    def compute_communication_time(self, total_batch_size: int) -> float:
        """计算通信时间（简化的 DDP 风格）"""
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        
        if self.config.num_workers <= 1:
            return 0.0
        
        # Ring All-Reduce 通信时间
        comm_time = 2 * msg_size / self.config.bandwidth_mb_s * math.log2(self.config.num_workers)
        base_latency = self.config.latency * math.log2(self.config.num_workers)
        
        return comm_time + base_latency
    
    def simulate_static_sharding(self) -> Dict:
        """
        静态分片策略：每个 worker 处理固定 batch size
        
        Returns:
            results: 模拟结果
        """
        # 每个 worker 使用相同的 batch size
        batch_size_per_worker = 64
        total_batch_size = batch_size_per_worker * self.config.num_workers
        
        # 计算每个 worker 的完成时间
        worker_times = []
        for worker in self.workers:
            compute_time = self.compute_worker_time(worker, batch_size_per_worker)
            worker_times.append(compute_time)
        
        # 整体 step 时间 = 最慢 worker 的计算时间 + 通信时间
        max_compute_time = max(worker_times)
        comm_time = self.compute_communication_time(total_batch_size)
        step_time = max_compute_time + comm_time
        
        # 计算吞吐量
        throughput = total_batch_size / step_time
        
        # 计算闲置时间
        idle_times = [(max_compute_time - t) for t in worker_times]
        avg_idle_time = sum(idle_times) / len(idle_times)
        
        return {
            'strategy': 'static_sharding',
            'throughput': throughput,
            'step_time': step_time,
            'max_compute_time': max_compute_time,
            'comm_time': comm_time,
            'worker_times': worker_times,
            'idle_times': idle_times,
            'avg_idle_time': avg_idle_time,
            'worker_speeds': [w.compute_speed for w in self.workers]
        }
    
    def simulate_dynamic_load_balancing(self) -> Dict:
        """
        动态负载均衡策略：根据 worker 速度动态分配 batch size
        
        目标：使所有 worker 同时完成计算
        """
        # 计算每个 worker 的速度权重
        total_speed = sum(w.compute_speed for w in self.workers)
        target_total_batch = 64 * self.config.num_workers  # 目标总 batch size
        
        # 根据速度比例分配 batch size
        allocated_batches = []
        for worker in self.workers:
            # 速度越快，分配越多 batch
            batch_size = max(1, int(target_total_batch * (worker.compute_speed / total_speed)))
            worker.batch_size = batch_size
            allocated_batches.append(batch_size)
        
        # 确保总 batch 数接近目标
        while sum(allocated_batches) < target_total_batch:
            # 找到速度最快的 worker，多分配一个 batch
            max_speed_idx = max(range(len(self.workers)), key=lambda i: self.workers[i].compute_speed)
            allocated_batches[max_speed_idx] += 1
        
        total_batch_size = sum(allocated_batches)
        
        # 计算每个 worker 的完成时间（应该接近相等）
        worker_times = []
        for i, worker in enumerate(self.workers):
            compute_time = self.compute_worker_time(worker, allocated_batches[i])
            worker_times.append(compute_time)
        
        # 整体 step 时间
        max_compute_time = max(worker_times)
        comm_time = self.compute_communication_time(total_batch_size)
        step_time = max_compute_time + comm_time
        
        # 计算吞吐量
        throughput = total_batch_size / step_time
        
        # 计算闲置时间（负载均衡后应该很小）
        idle_times = [(max_compute_time - t) for t in worker_times]
        avg_idle_time = sum(idle_times) / len(idle_times)
        
        return {
            'strategy': 'dynamic_load_balancing',
            'throughput': throughput,
            'step_time': step_time,
            'max_compute_time': max_compute_time,
            'comm_time': comm_time,
            'worker_times': worker_times,
            'idle_times': idle_times,
            'avg_idle_time': avg_idle_time,
            'allocated_batches': allocated_batches,
            'worker_speeds': [w.compute_speed for w in self.workers]
        }
    
    def run_comparison(self) -> Dict:
        """运行两种策略的对比"""
        static_result = self.simulate_static_sharding()
        dynamic_result = self.simulate_dynamic_load_balancing()
        
        return {
            'config': {
                'num_workers': self.config.num_workers,
                'speed_distribution': self.config.speed_distribution,
                'model_size_mb': self.config.model_size_mb,
                'bandwidth_mb_s': self.config.bandwidth_mb_s
            },
            'static_sharding': static_result,
            'dynamic_load_balancing': dynamic_result,
            'improvement': {
                'throughput': (dynamic_result['throughput'] - static_result['throughput']) / static_result['throughput'] * 100,
                'avg_idle_time': (static_result['avg_idle_time'] - dynamic_result['avg_idle_time']) / static_result['avg_idle_time'] * 100
            }
        }


def print_results(results: Dict):
    """打印对比结果"""
    print("=" * 80)
    print("Heterogeneous Worker Load Balancing Comparison")
    print("=" * 80)
    print(f"Number of Workers: {results['config']['num_workers']}")
    print(f"Speed Distribution: {results['config']['speed_distribution']}")
    print(f"Model Size: {results['config']['model_size_mb']} MB")
    print(f"Bandwidth: {results['config']['bandwidth_mb_s']} MB/s")
    print("=" * 80)
    
    # 打印策略对比
    static = results['static_sharding']
    dynamic = results['dynamic_load_balancing']
    
    print("\n--- Static Sharding ---")
    print(f"Throughput: {static['throughput']:.2f} samples/sec")
    print(f"Step Time: {static['step_time']:.4f} s")
    print(f"Average Idle Time: {static['avg_idle_time']:.4f} s")
    print(f"Worker Speeds: {[f'{s:.2f}x' for s in static['worker_speeds']]}")
    print(f"Worker Times: {[f'{t:.4f}s' for t in static['worker_times']]}")
    
    print("\n--- Dynamic Load Balancing ---")
    print(f"Throughput: {dynamic['throughput']:.2f} samples/sec")
    print(f"Step Time: {dynamic['step_time']:.4f} s")
    print(f"Average Idle Time: {dynamic['avg_idle_time']:.4f} s")
    print(f"Worker Speeds: {[f'{s:.2f}x' for s in dynamic['worker_speeds']]}")
    print(f"Allocated Batches: {dynamic['allocated_batches']}")
    print(f"Worker Times: {[f'{t:.4f}s' for t in dynamic['worker_times']]}")
    
    print("\n--- Improvement ---")
    print(f"Throughput Increase: {results['improvement']['throughput']:.1f}%")
    print(f"Idle Time Reduction: {results['improvement']['avg_idle_time']:.1f}%")


def plot_results(results: Dict):
    """绘制对比图表"""
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not installed, skipping plot generation")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 吞吐量对比
    strategies = ['Static Sharding', 'Dynamic Load Balancing']
    throughputs = [
        results['static_sharding']['throughput'],
        results['dynamic_load_balancing']['throughput']
    ]
    
    ax1.bar(strategies, throughputs, color=['blue', 'green'])
    ax1.set_ylabel('Throughput (samples/sec)')
    ax1.set_title('Throughput Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Worker 完成时间分布
    worker_ids = [f'Worker {i+1}' for i in range(results['config']['num_workers'])]
    static_times = results['static_sharding']['worker_times']
    dynamic_times = results['dynamic_load_balancing']['worker_times']
    
    x = range(len(worker_ids))
    width = 0.35
    
    ax2.bar([i - width/2 for i in x], static_times, width, label='Static Sharding', color='blue')
    ax2.bar([i + width/2 for i in x], dynamic_times, width, label='Dynamic Load Balancing', color='green')
    ax2.set_xlabel('Workers')
    ax2.set_ylabel('Compute Time (s)')
    ax2.set_title('Worker Compute Time Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(worker_ids)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heterogeneous_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to heterogeneous_results.png")


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous Worker Benchmark")
    
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--model-size", type=float, default=40.0, help="Model size (MB)")
    parser.add_argument("--bandwidth", type=float, default=100.0, help="Bandwidth (MB/s)")
    parser.add_argument("--distribution", type=str, default="random", 
                        choices=["random", "uniform", "tiered"],
                        help="Worker speed distribution")
    parser.add_argument("--min-speed", type=float, default=0.5, help="Minimum worker speed")
    parser.add_argument("--max-speed", type=float, default=2.0, help="Maximum worker speed")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # 创建配置
    config = HeterogeneousConfig(
        num_workers=args.workers,
        model_size_mb=args.model_size,
        bandwidth_mb_s=args.bandwidth,
        speed_distribution=args.distribution,
        min_speed=args.min_speed,
        max_speed=args.max_speed
    )
    
    # 运行模拟
    simulator = HeterogeneousSimulator(config)
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
