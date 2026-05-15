"""
Adaptive Controller

Central controller that coordinates all adaptive optimizations.
Monitors system metrics and triggers optimizations as needed.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional

from hermes_unified.optim.load_balancer import LoadBalancer
from hermes_unified.optim.overlap_scheduler import OverlapScheduler
from hermes_unified.optim.dynamic_compressor import DynamicCompressor
from hermes_unified.optim.protocol_switcher import ProtocolSwitcher, ProtocolEvaluator


class AdaptiveController:
    """
    Adaptive controller for distributed training optimization.
    
    Key features:
    - Periodically evaluates system performance
    - Coordinates load balancing, compression, overlap, and protocol switching
    - Maintains overall system state
    - Provides unified interface for optimization decisions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize adaptive controller.
        
        Args:
            config: Dictionary of optimization settings
        """
        self.config = config or {}
        
        # Initialize optimization modules
        self.load_balancer = LoadBalancer(
            min_batch_size=self.config.get('min_batch_size', 8),
            max_batch_size=self.config.get('max_batch_size', 256),
            ema_alpha=self.config.get('ema_alpha', 0.1)
        )
        
        self.overlap_scheduler = OverlapScheduler(
            num_segments=self.config.get('num_segments', 5)
        )
        
        self.compression_controller = DynamicCompressor(
            initial_ratio=self.config.get('initial_compression_ratio', 0.1),
            compression_type=self.config.get('compression_type', 'topk')
        )
        
        self.protocol_switcher = ProtocolSwitcher()
        self.protocol_evaluator = ProtocolEvaluator()
        
        # Monitoring
        self.metrics_history = deque(maxlen=100)
        self.evaluation_interval = self.config.get('evaluation_interval', 10)
        self.step_count = 0
        
        # State
        self.enabled = True
    
    def step(self, worker_times: Dict[int, float], worker_ids: List[int],
             model_params: np.ndarray, metrics: Optional[Dict] = None):
        """
        Execute one step of adaptive control.
        
        Args:
            worker_times: Computation times for each worker
            worker_ids: List of active worker IDs
            model_params: Current model parameters
            metrics: Optional additional metrics
        
        Returns:
            Dict of optimization decisions
        """
        self.step_count += 1
        
        decisions = {}
        
        # Periodic evaluation
        if self.step_count % self.evaluation_interval == 0 and self.enabled:
            decisions.update(self._evaluate_and_adapt(worker_times, worker_ids, model_params, metrics))
        
        return decisions
    
    def _evaluate_and_adapt(self, worker_times: Dict[int, float], worker_ids: List[int],
                            model_params: np.ndarray, metrics: Optional[Dict]) -> Dict:
        """
        Evaluate system state and make adaptation decisions.
        
        Returns:
            Dict containing optimization decisions
        """
        decisions = {}
        
        # 1. Update load balancer and compute new batch sizes
        self.load_balancer.update_worker_speeds(worker_times)
        total_batch = self.config.get('total_batch_size', 256)
        batch_sizes = self.load_balancer.compute_batch_sizes(total_batch, worker_ids)
        
        if batch_sizes:
            decisions['batch_sizes'] = batch_sizes
        
        # 2. Update compression based on bandwidth
        if metrics and 'bandwidth' in metrics:
            self.compression_controller.update_congestion(metrics['bandwidth'])
            decisions['compression_ratio'] = self.compression_controller.comp_ratio
        
        # 3. Evaluate and possibly switch protocol
        if metrics:
            best_protocol, protocol_times = self._evaluate_protocols(worker_ids, metrics)
            
            if best_protocol != self.protocol_switcher.current_protocol:
                success = self.protocol_switcher.switch_to(
                    best_protocol, worker_ids, model_params
                )
                if success:
                    decisions['protocol_switch'] = {
                        'from': self.protocol_switcher.current_protocol,
                        'to': best_protocol
                    }
        
        # Record metrics
        self.metrics_history.append({
            'step': self.step_count,
            'num_workers': len(worker_ids),
            'timestamp': time.time(),
            'decisions': decisions
        })
        
        return decisions
    
    def _evaluate_protocols(self, worker_ids: List[int], metrics: Dict) -> tuple:
        """
        Evaluate protocols and select best one.
        
        Returns:
            Tuple of (best_protocol, protocol_times)
        """
        num_workers = len(worker_ids)
        avg_grad_delay = metrics.get('avg_grad_delay', 0.1)
        bandwidth = metrics.get('bandwidth', 100.0)
        model_size_mb = metrics.get('model_size_mb', 40.0)
        
        return self.protocol_evaluator.select_best_protocol(
            num_workers, avg_grad_delay, bandwidth, model_size_mb
        )
    
    def get_load_balancer(self) -> LoadBalancer:
        """Get load balancer instance."""
        return self.load_balancer
    
    def get_overlap_scheduler(self) -> OverlapScheduler:
        """Get overlap scheduler instance."""
        return self.overlap_scheduler
    
    def get_compressor(self) -> DynamicCompressor:
        """Get dynamic compressor instance."""
        return self.compression_controller
    
    def get_protocol_switcher(self) -> ProtocolSwitcher:
        """Get protocol switcher instance."""
        return self.protocol_switcher
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            'step_count': self.step_count,
            'load_balancer_stats': {
                'speeds': self.load_balancer.worker_speeds,
                'history_size': sum(len(v) for v in self.load_balancer.worker_time_history.values())
            },
            'compressor_stats': self.compression_controller.get_stats(),
            'protocol_stats': self.protocol_switcher.get_switch_stats(),
            'metrics_history_length': len(self.metrics_history)
        }
    
    def enable(self):
        """Enable adaptive control."""
        self.enabled = True
    
    def disable(self):
        """Disable adaptive control."""
        self.enabled = False
    
    def reset(self):
        """Reset all optimization modules."""
        self.load_balancer.reset()
        self.compression_controller.reset_error()
        self.step_count = 0
        self.metrics_history.clear()
