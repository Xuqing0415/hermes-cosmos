#!/usr/bin/env python3
"""
Protocol Switch Module

实现思路二：自适应通信协议切换
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .core import SimulationConfig


@dataclass
class ProtocolSwitchConfig(SimulationConfig):
    """自适应协议切换配置"""
    initial_workers: int = 4
    max_workers: int = 32
    min_workers: int = 4
    switch_interval: int = 10  # 每多少步评估一次并可能切换
    worker_change_interval: int = 20  # 每多少步改变worker数量
    worker_change_amount: int = 4  # 每次worker数量变化量
    simulation_steps: int = 100
    switch_overhead: float = 0.1  # 切换开销（秒）


class ProtocolSwitchSimulator:
    """自适应通信协议模拟器"""
    
    def __init__(self, config: ProtocolSwitchConfig):
        self.config = config
        self.current_workers = config.initial_workers
        self.current_protocol = 'ddp'  # 初始协议
        self.switch_count = 0
        self.total_switch_overhead = 0.0
    
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
    
    def evaluate_protocol(self, protocol: str) -> Dict:
        """评估指定协议的性能"""
        if protocol == 'ps':
            comm_time = self.compute_ps_comm_time()
        elif protocol == 'ddp':
            comm_time = self.compute_ddp_comm_time()
        elif protocol == 'gossip':
            comm_time = self.compute_gossip_comm_time()
        else:
            comm_time = self.compute_ddp_comm_time()
        
        step_time = self.config.compute_time_per_worker + comm_time
        throughput = self.current_workers * self.config.batch_size_per_worker / step_time
        
        return {
            'protocol': protocol,
            'comm_time': comm_time,
            'step_time': step_time,
            'throughput': throughput
        }
    
    def select_best_protocol(self) -> str:
        """选择当前条件下最优的协议"""
        protocols = ['ps', 'ddp', 'gossip']
        throughputs = {p: self.evaluate_protocol(p)['throughput'] for p in protocols}
        
        # 选择吞吐量最高的协议
        best_protocol = max(throughputs, key=throughputs.get)
        return best_protocol, throughputs
    
    def simulate_adaptive_switching(self) -> List[Dict]:
        """模拟自适应协议切换"""
        results = []
        protocol_counts = {'ps': 0, 'ddp': 0, 'gossip': 0}
        switch_log = []
        
        for step in range(self.config.simulation_steps):
            # 周期性改变worker数量（模拟弹性伸缩）
            if step > 0 and step % self.config.worker_change_interval == 0:
                if self.current_workers >= self.config.max_workers:
                    # 减少worker
                    self.current_workers = max(self.config.min_workers, self.current_workers - self.config.worker_change_amount)
                else:
                    # 增加worker
                    self.current_workers = min(self.config.max_workers, self.current_workers + self.config.worker_change_amount)
            
            # 周期性评估并可能切换协议
            switch_made = False
            overhead = 0.0
            
            if step % self.config.switch_interval == 0:
                best_protocol, all_throughputs = self.select_best_protocol()
                
                if best_protocol != self.current_protocol:
                    # 记录切换
                    switch_log.append({
                        'step': step,
                        'from': self.current_protocol,
                        'to': best_protocol,
                        'throughputs': all_throughputs
                    })
                    
                    self.switch_count += 1
                    overhead = self.config.switch_overhead
                    self.total_switch_overhead += overhead
                    switch_made = True
                    
                    self.current_protocol = best_protocol
            
            # 计算当前协议的吞吐量
            current_result = self.evaluate_protocol(self.current_protocol)
            protocol_counts[self.current_protocol] += 1
            
            # 添加切换开销
            step_time_with_overhead = current_result['step_time'] + (overhead if switch_made else 0)
            throughput_with_overhead = self.current_workers * self.config.batch_size_per_worker / step_time_with_overhead
            
            results.append({
                'step': step,
                'num_workers': self.current_workers,
                'protocol': self.current_protocol,
                'throughput': throughput_with_overhead,
                'step_time': step_time_with_overhead,
                'switch_made': switch_made,
                'switch_overhead': overhead
            })
        
        return results, protocol_counts, switch_log
    
    def run_simulation(self) -> Dict:
        """运行完整模拟"""
        results, protocol_counts, switch_log = self.simulate_adaptive_switching()
        
        # 计算统计数据
        throughputs = [r['throughput'] for r in results]
        workers_list = [r['num_workers'] for r in results]
        
        return {
            'config': {
                'initial_workers': self.config.initial_workers,
                'max_workers': self.config.max_workers,
                'min_workers': self.config.min_workers,
                'model_size_mb': self.config.model_size_mb,
                'bandwidth': self.config.bandwidth_mb_s,
                'switch_interval': self.config.switch_interval,
                'switch_overhead': self.config.switch_overhead
            },
            'results': results,
            'protocol_counts': protocol_counts,
            'switch_log': switch_log,
            'statistics': {
                'avg_throughput': sum(throughputs) / len(throughputs),
                'min_throughput': min(throughputs),
                'max_throughput': max(throughputs),
                'avg_workers': sum(workers_list) / len(workers_list),
                'total_switches': self.switch_count,
                'total_switch_overhead': self.total_switch_overhead,
                'overhead_ratio': self.total_switch_overhead / (len(results) * self.config.compute_time_per_worker) * 100
            }
        }
