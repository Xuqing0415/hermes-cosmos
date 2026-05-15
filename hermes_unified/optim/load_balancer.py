"""
Dynamic Load Balancer

Implements proportional allocation algorithm for heterogeneous workers.
Uses Exponential Moving Average (EMA) to smooth historical computation times.
"""

import numpy as np
from collections import deque
from typing import Dict, List


class LoadBalancer:
    """
    Dynamic load balancer for heterogeneous workers.
    
    Key features:
    - Maintains EMA-smoothed computation speeds for each worker
    - Computes proportional batch sizes based on worker speeds
    - Enforces minimum and maximum batch size constraints
    - Smoothly transitions batch sizes to avoid abrupt changes
    """
    
    def __init__(self, 
                 min_batch_size: int = 8, 
                 max_batch_size: int = 256, 
                 ema_alpha: float = 0.1):
        """
        Initialize load balancer.
        
        Args:
            min_batch_size: Minimum batch size per worker
            max_batch_size: Maximum batch size per worker
            ema_alpha: EMA smoothing factor (0 < alpha <= 1)
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.ema_alpha = ema_alpha
        
        # Worker speed history (EMA-smoothed computation time inverse)
        self.worker_speeds: Dict[int, float] = {}
        self.worker_time_history: Dict[int, deque] = {}
        
    def update_worker_speeds(self, worker_times: Dict[int, float]):
        """
        Update worker speeds based on latest computation times.
        
        Args:
            worker_times: Dict mapping worker_id to computation time (seconds)
        """
        for worker_id, comp_time in worker_times.items():
            # Initialize if first time seeing this worker
            if worker_id not in self.worker_time_history:
                self.worker_time_history[worker_id] = deque(maxlen=20)
                self.worker_speeds[worker_id] = 1.0 / comp_time if comp_time > 0 else 1.0
            
            # Store raw time
            self.worker_time_history[worker_id].append(comp_time)
            
            # Compute EMA-smoothed speed
            speed = 1.0 / comp_time if comp_time > 0 else 1e-6
            self.worker_speeds[worker_id] = (
                self.ema_alpha * speed + 
                (1 - self.ema_alpha) * self.worker_speeds[worker_id]
            )
    
    def compute_batch_sizes(self, total_batch: int, worker_ids: List[int]) -> Dict[int, int]:
        """
        Compute proportional batch sizes for each worker.
        
        Args:
            total_batch: Total batch size across all workers
            worker_ids: List of active worker IDs
            
        Returns:
            Dict mapping worker_id to assigned batch size
        """
        if not worker_ids:
            return {}
        
        # Get speeds for active workers
        speeds = []
        for wid in worker_ids:
            speeds.append(self.worker_speeds.get(wid, 1.0))
        
        # Proportional allocation
        total_speed = sum(speeds)
        if total_speed == 0:
            total_speed = len(speeds)
            speeds = [1.0] * len(speeds)
        
        batch_sizes = {}
        for i, wid in enumerate(worker_ids):
            ratio = speeds[i] / total_speed
            batch_size = int(total_batch * ratio)
            batch_size = max(self.min_batch_size, min(self.max_batch_size, batch_size))
            batch_sizes[wid] = batch_size
        
        # Adjust to meet total batch size constraint
        current_total = sum(batch_sizes.values())
        diff = total_batch - current_total
        
        while diff != 0:
            for wid in worker_ids:
                if diff > 0 and batch_sizes[wid] < self.max_batch_size:
                    batch_sizes[wid] += 1
                    diff -= 1
                elif diff < 0 and batch_sizes[wid] > self.min_batch_size:
                    batch_sizes[wid] -= 1
                    diff += 1
                if diff == 0:
                    break
        
        return batch_sizes
    
    def get_worker_stats(self, worker_id: int) -> Dict:
        """Get statistics for a specific worker."""
        times = self.worker_time_history.get(worker_id, [])
        return {
            'speed': self.worker_speeds.get(worker_id, 1.0),
            'avg_time': np.mean(list(times)) if times else 0.0,
            'var_time': np.var(list(times)) if len(times) > 1 else 0.0,
            'sample_count': len(times)
        }
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.worker_speeds.clear()
        self.worker_time_history.clear()
