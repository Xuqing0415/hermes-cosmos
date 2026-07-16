#!/usr/bin/env python3
"""
Communication-Compute Overlap Module


"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .core import SimulationConfig


@dataclass
class OverlapConfig(SimulationConfig):
    """"""
    num_layers: int = 10  # 
    segment_count: int = 5  # 
    enable_adaptive: bool = True  # 


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
        """"""
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 
        step_time = compute_time + comm_time
        throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
        
        return {
            'overlap_factor': 0.0,
            'step_time': step_time,
            'throughput': throughput,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'segment_count': 1,
            'description': ''
        }
    
    def simulate_ideal_overlap(self) -> Dict:
        """"""
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 
        step_time = max(compute_time, comm_time)
        throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
        
        return {
            'overlap_factor': 1.0,
            'step_time': step_time,
            'throughput': throughput,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'segment_count': self.config.num_layers,
            'description': ''
        }
    
    def simulate_partial_overlap(self, overlap_factor: float) -> Dict:
        """
        
        
        Args:
            overlap_factor: 0~1
        """
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 
        num_segments = max(2, self.config.segment_count)
        segment_compute_time = compute_time / num_segments
        
        # 
        total_time = segment_compute_time  # 
        
        # 
        for _ in range(num_segments - 1):
            segment_comm_time = (comm_time / num_segments) * (1 - overlap_factor)
            total_time += max(segment_compute_time, segment_comm_time)
        
        # 
        remaining_comm_time = comm_time * overlap_factor
        total_time += remaining_comm_time
        
        throughput = self.config.num_workers * self.config.batch_size_per_worker / total_time
        
        return {
            'overlap_factor': overlap_factor,
            'step_time': total_time,
            'throughput': throughput,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'segment_count': num_segments,
            'description': f' ({overlap_factor*100:.0f}%)'
        }
    
    def simulate_dynamic_overlap(self) -> Dict:
        """
        
        """
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # /
        total_time = compute_time + comm_time
        comm_ratio = comm_time / total_time if total_time > 0 else 0.5
        
        # 
        # 
        dynamic_overlap = min(1.0, max(0.0, 2 * comm_ratio - 0.3))
        
        result = self.simulate_partial_overlap(dynamic_overlap)
        result['description'] = f' ({dynamic_overlap*100:.0f}%)'
        
        return result
    
    def simulate_adaptive_segmentation(self) -> Dict:
        """
        
        """
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 
        # 
        comm_ratio = comm_time / (compute_time + comm_time) if compute_time + comm_time > 0 else 0.5
        
        # 
        optimal_segments = max(2, min(self.config.num_layers, int(2 + 8 * comm_ratio)))
        
        # 
        original_segments = self.config.segment_count
        self.config.segment_count = optimal_segments
        
        result = self.simulate_dynamic_overlap()
        result['segment_count'] = optimal_segments
        result['description'] = f' ({optimal_segments})'
        
        # 
        self.config.segment_count = original_segments
        
        return result
    
    def run_overlap_study(self, overlap_factors: Optional[List[float]] = None) -> Dict:
        """"""
        if overlap_factors is None:
            overlap_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        results = {
            'config': {
                'num_workers': self.config.num_workers,
                'model_size_mb': self.config.model_size_mb,
                'bandwidth_mb_s': self.config.bandwidth_mb_s,
                'num_layers': self.config.num_layers
            },
            'fixed_overlap': [],
            'dynamic_overlap': None,
            'adaptive_segmentation': None
        }
        
        # 
        for factor in overlap_factors:
            result = self.simulate_partial_overlap(factor)
            results['fixed_overlap'].append(result)
        
        # 
        results['dynamic_overlap'] = self.simulate_dynamic_overlap()
        
        # 
        results['adaptive_segmentation'] = self.simulate_adaptive_segmentation()
        
        return results
