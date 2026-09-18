"""
Hardware-Aware Scheduler for Heterogeneous Workers
异构硬件感知调度器
"""

from typing import List, Dict, Tuple
import random

class WorkerInfo:
    """Worker信息"""
    
    def __init__(self, worker_id: int, device_type: str, speed: float):
        """
        Args:
            worker_id: Worker唯一标识
            device_type: 'cpu' 或 'gpu'
            speed: 计算速度（GFLOPS）
        """
        self.worker_id = worker_id
        self.device_type = device_type
        self.speed = speed
        self.busy = False
    
    def __repr__(self):
        return f"Worker({self.worker_id}, {self.device_type}, {self.speed} GFLOPS)"

class HardwareAwareScheduler:
    """硬件感知调度器"""
    
    def __init__(self, workers: List[WorkerInfo]):
        """
        Args:
            workers: Worker列表
        """
        self.workers = workers
        print(f"🚀 HardwareAwareScheduler initialized with {len(workers)} workers")
    
    def compatibility_score(self, trial_spec: Dict, worker: WorkerInfo) -> float:
        """计算trial与worker的兼容性分数"""
        score = 0.0
        
        # 大batch优先给GPU
        if worker.device_type == 'gpu' and trial_spec.get('batch_size', 32) > 64:
            score += 10.0
        
        # 高压缩率适合慢速CPU
        if worker.device_type == 'cpu' and trial_spec.get('compression', 0.1) > 0.3:
            score += 8.0
        
        # GPU速度加成
        if worker.device_type == 'gpu':
            score += worker.speed / 10.0
        else:
            score += worker.speed / 50.0
        
        # 学习率考虑（小学习率需要更多迭代，适合慢速设备）
        lr = trial_spec.get('lr', 0.01)
        if lr < 0.01 and worker.device_type == 'cpu':
            score += 5.0
        
        return score
    
    def assign_trial(self, trial_spec: Dict) -> WorkerInfo:
        """为trial分配最合适的worker"""
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
        """释放worker"""
        for worker in self.workers:
            if worker.worker_id == worker_id:
                worker.busy = False
                break
    
    def get_worker_stats(self) -> Dict:
        """获取worker统计信息"""
        stats = {
            'total': len(self.workers),
            'gpu_count': sum(1 for w in self.workers if w.device_type == 'gpu'),
            'cpu_count': sum(1 for w in self.workers if w.device_type == 'cpu'),
            'avg_gpu_speed': sum(w.speed for w in self.workers if w.device_type == 'gpu') / max(sum(1 for w in self.workers if w.device_type == 'gpu'), 1),
            'avg_cpu_speed': sum(w.speed for w in self.workers if w.device_type == 'cpu') / max(sum(1 for w in self.workers if w.device_type == 'cpu'), 1),
            'busy_count': sum(1 for w in self.workers if w.busy)
        }
        return stats