"""
Hardware-Aware Scheduler for Heterogeneous Workers

"""

from typing import List, Dict, Tuple
import random

class WorkerInfo:
    """Worker"""
    
    def __init__(self, worker_id: int, device_type: str, speed: float):
        """
        Args:
            worker_id: Worker
            device_type: 'cpu'  'gpu'
            speed: GFLOPS
        """
        self.worker_id = worker_id
        self.device_type = device_type
        self.speed = speed
        self.busy = False
    
    def __repr__(self):
        return f"Worker({self.worker_id}, {self.device_type}, {self.speed} GFLOPS)"

class HardwareAwareScheduler:
    """"""
    
    def __init__(self, workers: List[WorkerInfo]):
        """
        Args:
            workers: Worker
        """
        self.workers = workers
        print(f" HardwareAwareScheduler initialized with {len(workers)} workers")
    
    def compatibility_score(self, trial_spec: Dict, worker: WorkerInfo) -> float:
        """trialworker"""
        score = 0.0
        
        # batchGPU
        if worker.device_type == 'gpu' and trial_spec.get('batch_size', 32) > 64:
            score += 10.0
        
        # CPU
        if worker.device_type == 'cpu' and trial_spec.get('compression', 0.1) > 0.3:
            score += 8.0
        
        # GPU
        if worker.device_type == 'gpu':
            score += worker.speed / 10.0
        else:
            score += worker.speed / 50.0
        
        # 
        lr = trial_spec.get('lr', 0.01)
        if lr < 0.01 and worker.device_type == 'cpu':
            score += 5.0
        
        return score
    
    def assign_trial(self, trial_spec: Dict) -> WorkerInfo:
        """trialworker"""
        available_workers = [w for w in self.workers if not w.busy]
        
        if not available_workers:
            return random.choice(self.workers)
        
        best_worker = None
        best_score = -1
        
        for worker in available_workers:
            score = self.compatibility_score(trial_spec, worker)
            if score > best_score:
                best_score = score
                best_worker = worker
        
        if best_worker:
            best_worker.busy = True
        
        return best_worker
    
    def release_worker(self, worker_id: int):
        """worker"""
        for worker in self.workers:
            if worker.worker_id == worker_id:
                worker.busy = False
                break
    
    def get_worker_stats(self) -> Dict:
        """worker"""
        stats = {
            'total': len(self.workers),
            'gpu_count': sum(1 for w in self.workers if w.device_type == 'gpu'),
            'cpu_count': sum(1 for w in self.workers if w.device_type == 'cpu'),
            'avg_gpu_speed': sum(w.speed for w in self.workers if w.device_type == 'gpu') / max(sum(1 for w in self.workers if w.device_type == 'gpu'), 1),
            'avg_cpu_speed': sum(w.speed for w in self.workers if w.device_type == 'cpu') / max(sum(1 for w in self.workers if w.device_type == 'cpu'), 1),
            'busy_count': sum(1 for w in self.workers if w.busy)
        }
        return stats