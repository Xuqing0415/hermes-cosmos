"""
Communication-Computation Overlap Scheduler

Implements pipelined execution where gradient computation and communication
can overlap. Gradients are computed in segments and sent asynchronously.
"""

import threading
import time
import numpy as np
from typing import List, Dict, Callable


class OverlapScheduler:
    """
    Scheduler for overlapping computation and communication.
    
    Key features:
    - Splits model into segments for pipelined execution
    - Uses background threads for asynchronous gradient sending
    - Maintains completion tracking for synchronization
    - Supports configurable segment count
    """
    
    def __init__(self, num_segments: int = 5):
        """
        Initialize overlap scheduler.
        
        Args:
            num_segments: Number of segments to split the model into
        """
        self.num_segments = num_segments
        self.pending_sends = []
        self.send_lock = threading.Lock()
        self.send_condition = threading.Condition()
        self.completed_segments = set()
        self._running = True
        
        # Start background sender thread
        self._sender_thread = threading.Thread(target=self._background_sender, daemon=True)
        self._sender_thread.start()
    
    def segment_gradients(self, grad_list: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Split gradient list into segments for pipelined processing.
        
        Args:
            grad_list: List of gradients (one per layer)
            
        Returns:
            List of segments, each containing a subset of gradients
        """
        if not grad_list:
            return []
        
        n_layers = len(grad_list)
        segment_size = max(1, n_layers // self.num_segments)
        
        segments = []
        for i in range(self.num_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, n_layers)
            if start < end:
                segments.append(grad_list[start:end])
        
        return segments
    
    def send_segment(self, segment_idx: int, gradients: List[np.ndarray], 
                     send_fn: Callable[[List[np.ndarray]], None]):
        """
        Send a segment of gradients asynchronously.
        
        Args:
            segment_idx: Index of the segment being sent
            gradients: List of gradients in this segment
            send_fn: Function to send gradients (takes list of gradients)
        """
        with self.send_lock:
            self.pending_sends.append((segment_idx, gradients, send_fn))
        
        with self.send_condition:
            self.send_condition.notify()
    
    def _background_sender(self):
        """Background thread for sending gradients."""
        while self._running:
            with self.send_condition:
                self.send_condition.wait()
            
            while True:
                with self.send_lock:
                    if not self.pending_sends:
                        break
                    segment_idx, gradients, send_fn = self.pending_sends.pop(0)
                
                try:
                    send_fn(gradients)
                    with self.send_lock:
                        self.completed_segments.add(segment_idx)
                except Exception as e:
                    print(f"Error sending segment {segment_idx}: {e}")
    
    def wait_for_all(self, timeout: float = 30.0) -> bool:
        """
        Wait for all pending sends to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all completed, False if timed out
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.send_lock:
                if not self.pending_sends:
                    return True
            time.sleep(0.01)
        
        return False
    
    def is_segment_complete(self, segment_idx: int) -> bool:
        """Check if a specific segment has been sent."""
        with self.send_lock:
            return segment_idx in self.completed_segments
    
    def reset(self):
        """Reset state for new iteration."""
        with self.send_lock:
            self.pending_sends.clear()
            self.completed_segments.clear()
    
    def stop(self):
        """Stop the background sender thread."""
        self._running = False
        with self.send_condition:
            self.send_condition.notify()


class PipelineTimer:
    """
    Timer for measuring overlap efficiency.
    Tracks computation and communication times for each segment.
    """
    
    def __init__(self):
        self.timestamps = []
    
    def record_segment(self, segment_idx: int, comp_start: float, comp_end: float,
                       comm_start: float, comm_end: float):
        """Record timing for a segment."""
        self.timestamps.append({
            'segment': segment_idx,
            'comp_start': comp_start,
            'comp_end': comp_end,
            'comm_start': comm_start,
            'comm_end': comm_end,
            'comp_duration': comp_end - comp_start,
            'comm_duration': comm_end - comm_start,
            'overlap': max(0, min(comp_end, comm_end) - max(comp_start, comm_start))
        })
    
    def compute_overlap_ratio(self) -> float:
        """Compute overall overlap ratio."""
        if not self.timestamps:
            return 0.0
        
        total_comp = sum(t['comp_duration'] for t in self.timestamps)
        total_comm = sum(t['comm_duration'] for t in self.timestamps)
        total_overlap = sum(t['overlap'] for t in self.timestamps)
        
        if total_comm == 0:
            return 0.0
        
        return total_overlap / total_comm
    
    def get_total_time(self) -> float:
        """Get total time from first compute start to last comm end."""
        if not self.timestamps:
            return 0.0
        
        first_start = min(t['comp_start'] for t in self.timestamps)
        last_end = max(t['comm_end'] for t in self.timestamps)
        
        return last_end - first_start
