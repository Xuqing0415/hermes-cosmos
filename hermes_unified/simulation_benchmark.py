#!/usr/bin/env python3
"""
Hermes Unified Simulation Benchmark

纯 Python 模拟分布式训练性能，无需实际 GPU 或分布式环境。

核心功能：
1. 模拟 PS（参数服务器）、DDP（All-Reduce）、Gossip 三种模式
2. 计算通信时间、计算时间、吞吐量
3. 支持梯度压缩、计算通信重叠
4. 生成对比图表和报告
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
    """模拟配置参数"""
    num_workers_list: List[int] = None
    model_size_mb: float = 40.0  # 模型大小（MB），10M参数 * 4 bytes = 40MB
    compute_time_per_worker: float = 0.02  # 单个worker纯计算耗时（秒）
    bandwidth_mb_s: float = 100.0  # 网络带宽（MB/s）
    latency: float = 0.0001  # 网络延迟（秒）
    gossip_neighbors: int = 2  # Gossip模式每轮通信邻居数
    compression_ratio: float = 1.0  # 梯度压缩比例
    overlap_factor: float = 0.0  # 计算通信重叠因子（0=串行，1=完全重叠）
    batch_size_per_worker: int = 64  # 每个worker的batch size
    enable_noise: bool = False  # 是否添加随机噪声


class DistributedSimulator:
    """分布式训练模拟器"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def compute_communication_time_ps(self, num_workers: int) -> float:
        """
        计算参数服务器模式的通信时间
        
        PS模式：每个worker上传梯度，PS聚合后广播回所有worker
        """
        # 发送的消息大小（压缩后）
        msg_size_send = self.config.model_size_mb * self.config.compression_ratio
        # 接收的消息大小（聚合后的完整梯度）
        msg_size_recv = self.config.model_size_mb
        
        # 上传时间：所有worker并行上传
        upload_time = msg_size_send / self.config.bandwidth_mb_s
        # 下载时间：PS广播给所有worker（假设并行）
        download_time = msg_size_recv / self.config.bandwidth_mb_s
        
        # 加上基础延迟（每轮通信）
        base_latency = self.config.latency * 2  # 上传+下载
        
        return upload_time + download_time + base_latency
    
    def compute_communication_time_ddp(self, num_workers: int) -> float:
        """
        计算DDP模式的通信时间（Ring All-Reduce）
        
        Ring All-Reduce：总传输量 = 2 * (num_workers - 1) / num_workers * model_size
        简化公式：comm_time = 2 * model_size / bandwidth * log2(num_workers)
        """
        if num_workers <= 1:
            return 0.0
        
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        
        # Ring All-Reduce 通信时间
        # 每个节点需要发送 (num_workers-1) 次，接收 (num_workers-1) 次
        # 但由于是环形，可以并行，所以时间 = 2 * (num_workers-1) * msg_size / (num_workers * bandwidth)
        # 简化为：2 * msg_size / bandwidth * log2(num_workers)
        
        comm_time = 2 * msg_size / self.config.bandwidth_mb_s * math.log2(num_workers)
        
        # 加上基础延迟
        base_latency = self.config.latency * math.log2(num_workers)
        
        return comm_time + base_latency
    
    def compute_communication_time_gossip(self, num_workers: int) -> float:
        """
        计算Gossip模式的通信时间
        
        Gossip：每个worker与k个邻居交换参数
        """
        k = self.config.gossip_neighbors
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        
        # 每轮交换：发送model_size，接收model_size
        # 假设所有worker并行执行
        comm_time = k * (2 * msg_size / self.config.bandwidth_mb_s)
        
        # 加上基础延迟
        base_latency = self.config.latency * k
        
        return comm_time + base_latency
    
    def compute_step_time(self, num_workers: int, comm_time: float) -> float:
        """
        计算单个step的总时间
        
        考虑计算与通信的重叠：
        step_time = max(compute_time, comm_time * overlap) + (1 - overlap) * (compute_time + comm_time)
        """
        # 计算时间（所有worker并行，所以取单个worker的时间）
        compute_time = self.config.compute_time_per_worker
        
        # 应用重叠因子
        overlap = self.config.overlap_factor
        step_time = max(compute_time, comm_time * overlap) + (1 - overlap) * (compute_time + comm_time)
        
        # 添加随机噪声
        if self.config.enable_noise:
            noise = random.uniform(-0.05, 0.05)
            step_time *= (1 + noise)
        
        return max(step_time, 0.0001)  # 防止负数
    
    def compute_throughput(self, num_workers: int, step_time: float) -> float:
        """
        计算吞吐量（samples/sec）
        
        throughput = batch_size * num_workers / step_time
        """
        return self.config.batch_size_per_worker * num_workers / step_time
    
    def run_simulation(self) -> Dict[str, List[Dict]]:
        """
        运行完整模拟
        
        Returns:
            results: 包含各模式的结果列表
        """
        results = {
            'ps': [],
            'ddp': [],
            'gossip': []
        }
        
        for num_workers in self.config.num_workers_list:
            # 参数服务器模式
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
            
            # DDP模式
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
            
            # Gossip模式
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
    """打印模拟结果表格"""
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
    """绘制结果图表"""
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not installed, skipping plot generation")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 吞吐量图
    colors = {'ps': 'blue', 'ddp': 'green', 'gossip': 'purple'}
    labels = {'ps': 'PS (参数服务器)', 'ddp': 'DDP (All-Reduce)', 'gossip': 'Gossip'}
    
    for mode, data in results.items():
        workers = [e['num_workers'] for e in data]
        throughput = [e['throughput'] for e in data]
        ax1.plot(workers, throughput, marker='o', label=labels[mode], color=colors[mode])
    
    ax1.set_xlabel('Worker数量')
    ax1.set_ylabel('吞吐量 (samples/sec)')
    ax1.set_title('不同模式的吞吐量对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 通信占比图
    for mode, data in results.items():
        workers = [e['num_workers'] for e in data]
        comm_ratio = [e['comm_ratio'] for e in data]
        ax2.plot(workers, comm_ratio, marker='s', label=labels[mode], color=colors[mode])
    
    ax2.set_xlabel('Worker数量')
    ax2.set_ylabel('通信时间占比 (%)')
    ax2.set_title('通信时间占比随Worker数量变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 图表已保存到 simulation_results.png")


def run_simulation_with_config(config: SimulationConfig) -> Dict[str, List[Dict]]:
    """运行模拟并返回结果"""
    simulator = DistributedSimulator(config)
    results = simulator.run_simulation()
    
    print_results(results, config)
    plot_results(results, config)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Hermes Unified Simulation Benchmark")
    
    # 模型参数
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
    
    # 解析worker列表
    workers_list = [int(w.strip()) for w in args.workers.split(',')]
    
    # 创建配置
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
    
    # 运行模拟
    results = run_simulation_with_config(config)
    
    # 保存结果
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
        print(f"\n📄 结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
