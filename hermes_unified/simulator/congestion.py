#!/usr/bin/env python3
"""
Network Congestion Module


"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .core import SimulationConfig


@dataclass
class CongestionConfig(SimulationConfig):
    """"""
    congestion_probability: float = 0.2  # 
    congestion_factor: float = 0.5  # 
    initial_compression_ratio: float = 0.1  # 
    min_compression_ratio: float = 0.01  # 
    max_compression_ratio: float = 1.0  # 
    compression_adjust_factor: float = 0.1  # 
    smoothing_factor: float = 0.9  # 
    simulation_steps: int = 100  # 
    congestion_start_step: int = 50  # 
    congestion_end_step: int = 80  # 


class CongestionSimulator:
    """"""
    
    def __init__(self, config: CongestionConfig):
        self.config = config
        self.current_bandwidth = config.bandwidth_mb_s
        self.current_compression = config.initial_compression_ratio
        self.congested = False
        self.estimated_bandwidth = config.bandwidth_mb_s
        self.loss_history = []
    
    def compute_communication_time(self, compression_ratio: float) -> float:
        """"""
        if self.config.num_workers <= 1:
            return 0.0
        
        msg_size = self.config.model_size_mb * compression_ratio
        comm_time = 2 * msg_size / self.current_bandwidth * math.log2(self.config.num_workers)
        base_latency = self.config.latency * math.log2(self.config.num_workers)
        
        return comm_time + base_latency
    
    def detect_congestion(self, actual_comm_time: float, expected_comm_time: float) -> bool:
        """
        
        
        Args:
            actual_comm_time: 
            expected_comm_time: 
        
        Returns:
            congested: 
        """
        # 20%
        if actual_comm_time > expected_comm_time * 1.2:
            return True
        return False
    
    def adjust_compression_ratio(self, congested: bool) -> float:
        """
        
        
        Args:
            congested: 
        
        Returns:
            new_compression_ratio: 
        """
        if congested:
            # 
            new_ratio = max(
                self.config.min_compression_ratio,
                self.current_compression * (1 - self.config.compression_adjust_factor)
            )
        else:
            # 
            new_ratio = min(
                self.config.max_compression_ratio,
                self.current_compression * (1 + self.config.compression_adjust_factor * 0.5)
            )
        
        # 
        self.current_compression = self.config.smoothing_factor * self.current_compression + \
                                  (1 - self.config.smoothing_factor) * new_ratio
        
        return self.current_compression
    
    def simulate_static_compression(self) -> List[Dict]:
        """"""
        results = []
        
        for step in range(self.config.simulation_steps):
            # 
            if self.config.congestion_start_step <= step < self.config.congestion_end_step:
                self.current_bandwidth = self.config.bandwidth_mb_s * self.config.congestion_factor
                self.congested = True
            else:
                self.current_bandwidth = self.config.bandwidth_mb_s
                self.congested = False
            
            # 
            compression_ratio = self.config.initial_compression_ratio
            
            comm_time = self.compute_communication_time(compression_ratio)
            step_time = self.config.compute_time_per_worker + comm_time
            throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
            
            # 
            loss = 0.1 + (1 - compression_ratio) * 0.2
            
            results.append({
                'step': step,
                'bandwidth': self.current_bandwidth,
                'compression_ratio': compression_ratio,
                'congested': self.congested,
                'throughput': throughput,
                'step_time': step_time,
                'loss': loss,
                'strategy': 'static'
            })
        
        return results
    
    def simulate_dynamic_compression(self) -> List[Dict]:
        """"""
        results = []
        self.current_compression = self.config.initial_compression_ratio
        
        for step in range(self.config.simulation_steps):
            # 
            if self.config.congestion_start_step <= step < self.config.congestion_end_step:
                self.current_bandwidth = self.config.bandwidth_mb_s * self.config.congestion_factor
                self.congested = True
            else:
                self.current_bandwidth = self.config.bandwidth_mb_s
                self.congested = False
            
            # 
            self.adjust_compression_ratio(self.congested)
            compression_ratio = self.current_compression
            
            comm_time = self.compute_communication_time(compression_ratio)
            step_time = self.config.compute_time_per_worker + comm_time
            throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
            
            # 
            loss = 0.1 + (1 - compression_ratio) * 0.2
            
            results.append({
                'step': step,
                'bandwidth': self.current_bandwidth,
                'compression_ratio': compression_ratio,
                'congested': self.congested,
                'throughput': throughput,
                'step_time': step_time,
                'loss': loss,
                'strategy': 'dynamic'
            })
        
        return results
    
    def simulate_congestion_detection(self) -> List[Dict]:
        """"""
        results = []
        self.current_compression = self.config.initial_compression_ratio
        
        for step in range(self.config.simulation_steps):
            # 
            if random.random() < self.config.congestion_probability:
                self.current_bandwidth = self.config.bandwidth_mb_s * self.config.congestion_factor
            else:
                self.current_bandwidth = self.config.bandwidth_mb_s
            
            # 
            expected_comm_time = 2 * self.config.model_size_mb * self.current_compression / \
                                self.config.bandwidth_mb_s * math.log2(self.config.num_workers)
            
            # 
            comm_time = self.compute_communication_time(self.current_compression)
            
            # 
            self.congested = self.detect_congestion(comm_time, expected_comm_time)
            
            # 
            self.adjust_compression_ratio(self.congested)
            
            step_time = self.config.compute_time_per_worker + comm_time
            throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
            loss = 0.1 + (1 - self.current_compression) * 0.2
            
            results.append({
                'step': step,
                'bandwidth': self.current_bandwidth,
                'compression_ratio': self.current_compression,
                'congested': self.congested,
                'throughput': throughput,
                'step_time': step_time,
                'loss': loss,
                'strategy': 'detection_based'
            })
        
        return results
    
    def run_comparison(self) -> Dict:
        """ vs """
        static_results = self.simulate_static_compression()
        dynamic_results = self.simulate_dynamic_compression()
        
        # 
        static_throughputs = [r['throughput'] for r in static_results]
        dynamic_throughputs = [r['throughput'] for r in dynamic_results]
        static_losses = [r['loss'] for r in static_results]
        dynamic_losses = [r['loss'] for r in dynamic_results]
        
        return {
            'config': {
                'num_workers': self.config.num_workers,
                'model_size_mb': self.config.model_size_mb,
                'base_bandwidth': self.config.bandwidth_mb_s,
                'congestion_probability': self.config.congestion_probability,
                'initial_compression': self.config.initial_compression_ratio
            },
            'static_compression': static_results,
            'dynamic_compression': dynamic_results,
            'statistics': {
                'static_avg_throughput': sum(static_throughputs) / len(static_throughputs),
                'static_min_throughput': min(static_throughputs),
                'static_max_throughput': max(static_throughputs),
                'static_avg_loss': sum(static_losses) / len(static_losses),
                'dynamic_avg_throughput': sum(dynamic_throughputs) / len(dynamic_throughputs),
                'dynamic_min_throughput': min(dynamic_throughputs),
                'dynamic_max_throughput': max(dynamic_throughputs),
                'dynamic_avg_loss': sum(dynamic_losses) / len(dynamic_losses)
            }
        }
