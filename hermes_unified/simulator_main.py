#!/usr/bin/env python3
"""
Hermes Unified Simulator Main Entry

统一的模拟框架入口，支持四种分析模式：
1. heterogeneous - 异构 Worker 负载均衡
2. overlap - 通信与计算重叠优化
3. congestion - 网络拥塞与动态压缩
4. protocol_switch - 自适应通信协议切换
"""

import argparse
import json
import os

from hermes_unified.simulator import (
    HeterogeneousSimulator,
    HeterogeneousConfig,
    OverlapSimulator,
    OverlapConfig,
    CongestionSimulator,
    CongestionConfig,
    ProtocolSwitchSimulator,
    ProtocolSwitchConfig,
    plot_results
)


def run_heterogeneous_analysis(args):
    """运行异构负载均衡分析"""
    config = HeterogeneousConfig(
        num_workers=args.workers,
        speed_distribution=args.distribution,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        model_size_mb=args.model_size,
        bandwidth_mb_s=args.bandwidth
    )
    
    simulator = HeterogeneousSimulator(config)
    results = simulator.run_comparison()
    
    # 打印结果
    print("=" * 80)
    print("Heterogeneous Worker Load Balancing Analysis")
    print("=" * 80)
    print(f"Configuration: {results['config']}")
    print("\n--- Static Sharding ---")
    print(f"Throughput: {results['static_sharding']['throughput']:.2f} samples/sec")
    print(f"Avg Idle Time: {results['static_sharding']['avg_idle_time']:.4f} s")
    print(f"Worker Speeds: {[f'{s:.2f}x' for s in results['static_sharding']['worker_speeds']]}")
    
    print("\n--- Dynamic Load Balancing ---")
    print(f"Throughput: {results['dynamic_load_balancing']['throughput']:.2f} samples/sec")
    print(f"Avg Idle Time: {results['dynamic_load_balancing']['avg_idle_time']:.4f} s")
    print(f"Batch Sizes: {results['dynamic_load_balancing']['batch_sizes']}")
    
    print("\n--- Improvement ---")
    print(f"Throughput Increase: {results['improvement']['throughput']:.1f}%")
    print(f"Idle Time Reduction: {results['improvement']['avg_idle_time']:.1f}%")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # 绘制图表
    plot_results(results, 'heterogeneous', args.plot_dir)


def run_overlap_analysis(args):
    """运行通信与计算重叠分析"""
    config = OverlapConfig(
        num_workers=args.workers,
        model_size_mb=args.model_size,
        bandwidth_mb_s=args.bandwidth,
        num_layers=args.layers
    )
    
    simulator = OverlapSimulator(config)
    results = simulator.run_overlap_study()
    
    # 打印结果
    print("=" * 80)
    print("Communication-Compute Overlap Analysis")
    print("=" * 80)
    print(f"Configuration: {results['config']}")
    
    print("\n--- Fixed Overlap Factors ---")
    for result in results['fixed_overlap']:
        print(f"Overlap={result['overlap_factor']:.1f}: Throughput={result['throughput']:.2f} samples/sec")
    
    print("\n--- Dynamic Overlap ---")
    dynamic = results['dynamic_overlap']
    print(f"Optimal Overlap Factor: {dynamic['overlap_factor']:.3f}")
    print(f"Throughput: {dynamic['throughput']:.2f} samples/sec")
    
    print("\n--- Adaptive Segmentation ---")
    adaptive = results['adaptive_segmentation']
    print(f"Optimal Segments: {adaptive['segment_count']}")
    print(f"Throughput: {adaptive['throughput']:.2f} samples/sec")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # 绘制图表
    plot_results(results, 'overlap', args.plot_dir)


def run_congestion_analysis(args):
    """运行网络拥塞分析"""
    config = CongestionConfig(
        num_workers=args.workers,
        model_size_mb=args.model_size,
        bandwidth_mb_s=args.bandwidth,
        congestion_probability=args.congestion_prob,
        simulation_steps=args.steps
    )
    
    simulator = CongestionSimulator(config)
    results = simulator.run_comparison()
    
    # 打印结果
    print("=" * 80)
    print("Network Congestion Analysis")
    print("=" * 80)
    print(f"Configuration: {results['config']}")
    
    stats = results['statistics']
    print("\n--- Statistics ---")
    print(f"{'Metric':<30} {'Static':<15} {'Dynamic':<15}")
    print("-" * 60)
    print(f"{'Average Throughput':<30} {stats['static_avg_throughput']:<15.2f} {stats['dynamic_avg_throughput']:<15.2f}")
    print(f"{'Minimum Throughput':<30} {stats['static_min_throughput']:<15.2f} {stats['dynamic_min_throughput']:<15.2f}")
    print(f"{'Average Loss':<30} {stats['static_avg_loss']:<15.4f} {stats['dynamic_avg_loss']:<15.4f}")
    
    improvement = (stats['dynamic_avg_throughput'] - stats['static_avg_throughput']) / stats['static_avg_throughput'] * 100
    print(f"\nThroughput Improvement: {improvement:.1f}%")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # 绘制图表
    plot_results(results, 'congestion', args.plot_dir)


def run_protocol_switch_analysis(args):
    """运行自适应协议切换分析"""
    config = ProtocolSwitchConfig(
        initial_workers=args.initial_workers,
        max_workers=args.max_workers,
        min_workers=args.min_workers,
        model_size_mb=args.model_size,
        bandwidth_mb_s=args.bandwidth,
        simulation_steps=args.steps,
        switch_interval=args.switch_interval
    )
    
    simulator = ProtocolSwitchSimulator(config)
    results = simulator.run_simulation()
    
    # 打印结果
    print("=" * 80)
    print("Adaptive Protocol Switching Analysis")
    print("=" * 80)
    print(f"Configuration: {results['config']}")
    
    stats = results['statistics']
    print("\n--- Statistics ---")
    print(f"Average Throughput: {stats['avg_throughput']:.2f} samples/sec")
    print(f"Average Workers: {stats['avg_workers']:.1f}")
    print(f"Total Switches: {stats['total_switches']}")
    print(f"Total Switch Overhead: {stats['total_switch_overhead']:.2f} s")
    print(f"Overhead Ratio: {stats['overhead_ratio']:.2f}%")
    
    print("\n--- Protocol Usage ---")
    counts = results['protocol_counts']
    total = sum(counts.values())
    for proto, count in counts.items():
        print(f"{proto.upper()}: {count} steps ({count/total*100:.1f}%)")
    
    print("\n--- Switch Log ---")
    for switch in results['switch_log']:
        print(f"Step {switch['step']}: {switch['from'].upper()} → {switch['to'].upper()}")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # 绘制图表
    plot_results(results, 'protocol_switch', args.plot_dir)


def main():
    parser = argparse.ArgumentParser(description="Hermes Unified Simulator")
    subparsers = parser.add_subparsers(dest='analysis_type', help='Analysis type')
    
    # 异构负载均衡
    hetero_parser = subparsers.add_parser('heterogeneous', help='Heterogeneous worker load balancing')
    hetero_parser.add_argument('--workers', type=int, default=4)
    hetero_parser.add_argument('--distribution', type=str, default='tiered', choices=['random', 'uniform', 'tiered'])
    hetero_parser.add_argument('--min-speed', type=float, default=0.5)
    hetero_parser.add_argument('--max-speed', type=float, default=2.0)
    hetero_parser.add_argument('--model-size', type=float, default=40.0)
    hetero_parser.add_argument('--bandwidth', type=float, default=100.0)
    hetero_parser.add_argument('--output', type=str, default='')
    hetero_parser.add_argument('--plot-dir', type=str, default='results')
    
    # 通信与计算重叠
    overlap_parser = subparsers.add_parser('overlap', help='Communication-compute overlap')
    overlap_parser.add_argument('--workers', type=int, default=4)
    overlap_parser.add_argument('--model-size', type=float, default=40.0)
    overlap_parser.add_argument('--bandwidth', type=float, default=100.0)
    overlap_parser.add_argument('--layers', type=int, default=10)
    overlap_parser.add_argument('--output', type=str, default='')
    overlap_parser.add_argument('--plot-dir', type=str, default='results')
    
    # 网络拥塞
    congestion_parser = subparsers.add_parser('congestion', help='Network congestion')
    congestion_parser.add_argument('--workers', type=int, default=4)
    congestion_parser.add_argument('--model-size', type=float, default=40.0)
    congestion_parser.add_argument('--bandwidth', type=float, default=100.0)
    congestion_parser.add_argument('--congestion-prob', type=float, default=0.2)
    congestion_parser.add_argument('--steps', type=int, default=100)
    congestion_parser.add_argument('--output', type=str, default='')
    congestion_parser.add_argument('--plot-dir', type=str, default='results')
    
    # 自适应协议切换
    proto_parser = subparsers.add_parser('protocol-switch', help='Adaptive protocol switching')
    proto_parser.add_argument('--initial-workers', type=int, default=4)
    proto_parser.add_argument('--max-workers', type=int, default=32)
    proto_parser.add_argument('--min-workers', type=int, default=4)
    proto_parser.add_argument('--model-size', type=float, default=40.0)
    proto_parser.add_argument('--bandwidth', type=float, default=100.0)
    proto_parser.add_argument('--steps', type=int, default=100)
    proto_parser.add_argument('--switch-interval', type=int, default=10)
    proto_parser.add_argument('--output', type=str, default='')
    proto_parser.add_argument('--plot-dir', type=str, default='results')
    
    args = parser.parse_args()
    
    if args.analysis_type == 'heterogeneous':
        run_heterogeneous_analysis(args)
    elif args.analysis_type == 'overlap':
        run_overlap_analysis(args)
    elif args.analysis_type == 'congestion':
        run_congestion_analysis(args)
    elif args.analysis_type == 'protocol-switch':
        run_protocol_switch_analysis(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
