"""
Protocol Switcher

Implements protocol switching logic between PS, DDP, and Gossip modes.
Handles graceful transition including state save/restore.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional


class ProtocolSwitcher:
    """
    Protocol switcher for distributed training.
    
    Key features:
    - Supports three protocols: PS, DDP, GOSSIP
    - Handles graceful protocol transitions
    - Maintains state during transitions
    - Tracks switch overhead
    """
    
    PROTOCOLS = ['ps', 'ddp', 'gossip']
    
    def __init__(self):
        self.current_protocol = 'ps'
        self.switch_count = 0
        self.switch_history = []
        self.last_switch_time = 0.0
    
    def switch_to(self, new_protocol: str, workers: List[int], 
                  model_params: np.ndarray, switch_timeout: float = 10.0) -> bool:
        """
        Switch to a new protocol.
        
        Args:
            new_protocol: Target protocol ('ps', 'ddp', 'gossip')
            workers: List of active worker IDs
            model_params: Current model parameters to preserve
            switch_timeout: Maximum time allowed for switch
            
        Returns:
            True if switch successful, False otherwise
        """
        if new_protocol not in self.PROTOCOLS:
            print(f"Unknown protocol: {new_protocol}")
            return False
        
        if new_protocol == self.current_protocol:
            print(f"Already using {new_protocol}")
            return True
        
        start_time = time.time()
        print(f"Switching from {self.current_protocol} to {new_protocol}...")
        
        try:
            # Step 1: Save current state
            state = self._save_state(model_params)
            
            # Step 2: Stop current communication
            self._stop_current_protocol(workers)
            
            # Step 3: Initialize new protocol
            success = self._initialize_protocol(new_protocol, workers, state)
            
            if success:
                # Step 4: Restore parameters to new protocol
                self._restore_state(model_params)
                
                switch_time = time.time() - start_time
                self._record_switch(new_protocol, switch_time, success)
                self.current_protocol = new_protocol
                self.last_switch_time = time.time()
                
                print(f"Successfully switched to {new_protocol} (took {switch_time:.3f}s)")
                return True
            else:
                # Rollback on failure
                self._rollback(state)
                print(f"Failed to switch to {new_protocol}, rolled back")
                return False
                
        except Exception as e:
            print(f"Error during switch: {e}")
            return False
    
    def _save_state(self, model_params: np.ndarray) -> Dict:
        """Save current protocol state."""
        return {
            'protocol': self.current_protocol,
            'timestamp': time.time(),
            'params_checksum': np.sum(model_params)
        }
    
    def _stop_current_protocol(self, workers: List[int]):
        """Stop current protocol's communication."""
        print(f"Stopping {self.current_protocol} protocol for {len(workers)} workers")
        # In real implementation: send stop signals to all workers
    
    def _initialize_protocol(self, protocol: str, workers: List[int], state: Dict) -> bool:
        """Initialize new protocol."""
        print(f"Initializing {protocol} protocol for {len(workers)} workers")
        
        # Simulate initialization time
        init_time = {
            'ps': 0.1,
            'ddp': 0.3 + len(workers) * 0.02,
            'gossip': 0.5 + len(workers) * 0.01
        }
        
        time.sleep(init_time.get(protocol, 0.2))
        return True
    
    def _restore_state(self, model_params: np.ndarray):
        """Restore parameters to newly initialized protocol."""
        print("Restoring model parameters")
    
    def _rollback(self, state: Dict):
        """Rollback to previous state on failure."""
        print(f"Rolling back to {state['protocol']}")
    
    def _record_switch(self, protocol: str, duration: float, success: bool):
        """Record switch event."""
        self.switch_count += 1
        self.switch_history.append({
            'protocol': protocol,
            'timestamp': time.time(),
            'duration': duration,
            'success': success
        })
    
    def get_switch_stats(self) -> Dict:
        """Get statistics about protocol switches."""
        if not self.switch_history:
            return {
                'switch_count': 0,
                'avg_switch_time': 0.0,
                'last_protocol': self.current_protocol
            }
        
        successful_switches = [s for s in self.switch_history if s['success']]
        
        return {
            'switch_count': self.switch_count,
            'successful_switches': len(successful_switches),
            'avg_switch_time': np.mean([s['duration'] for s in successful_switches]) if successful_switches else 0.0,
            'last_protocol': self.current_protocol,
            'protocols_used': set(s['protocol'] for s in self.switch_history)
        }


class ProtocolEvaluator:
    """
    Evaluates performance of different protocols.
    
    Uses performance models to predict step time for each protocol.
    """
    
    def __init__(self):
        self.metrics_history = deque(maxlen=50)
    
    def evaluate_protocol(self, protocol: str, num_workers: int, 
                          avg_grad_delay: float, bandwidth: float,
                          model_size_mb: float) -> float:
        """
        Evaluate protocol performance.
        
        Args:
            protocol: Protocol to evaluate
            num_workers: Number of workers
            avg_grad_delay: Average gradient delay in seconds
            bandwidth: Network bandwidth in MB/s
            model_size_mb: Model size in MB
            
        Returns:
            Predicted step time in seconds
        """
        if num_workers <= 1:
            return avg_grad_delay  # No communication needed
        
        if protocol == 'ps':
            # PS: centralized, higher overhead with many workers
            base_overhead = 0.1 + num_workers * 0.02
            comm_time = (model_size_mb / bandwidth) * (1 + base_overhead)
            return max(avg_grad_delay, comm_time)
        
        elif protocol == 'ddp':
            # DDP: ring all-reduce, logarithmic scaling
            base_overhead = 0.05 + np.log2(num_workers) * 0.03
            comm_time = (model_size_mb / bandwidth) * np.log2(num_workers) * (1 + base_overhead)
            return max(avg_grad_delay, comm_time)
        
        elif protocol == 'gossip':
            # Gossip: decentralized, constant per-worker overhead
            base_overhead = 0.3 + num_workers * 0.01
            comm_time = (model_size_mb / bandwidth) * 2 * (1 + base_overhead)
            return max(avg_grad_delay, comm_time)
        
        else:
            return float('inf')
    
    def select_best_protocol(self, num_workers: int, avg_grad_delay: float,
                             bandwidth: float, model_size_mb: float) -> str:
        """Select protocol with minimal predicted step time."""
        protocols = ['ps', 'ddp', 'gossip']
        times = {}
        
        for protocol in protocols:
            times[protocol] = self.evaluate_protocol(
                protocol, num_workers, avg_grad_delay, bandwidth, model_size_mb
            )
        
        return min(times, key=times.get), times
    
    def update_metrics(self, num_workers: int, avg_grad_delay: float,
                       bandwidth: float, model_size_mb: float):
        """Record current metrics."""
        self.metrics_history.append({
            'num_workers': num_workers,
            'avg_grad_delay': avg_grad_delay,
            'bandwidth': bandwidth,
            'model_size_mb': model_size_mb,
            'timestamp': time.time()
        })
