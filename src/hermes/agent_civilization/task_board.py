import time
from typing import List, Dict, Optional, Any

from .types import Task, TaskStatus


class TaskBoard:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.pending_tasks: List[str] = []
        self.in_progress_tasks: List[str] = []
        self.completed_tasks: List[str] = []

    def publish_task(self, title: str, description: str, task_type: str,
                     input_data: Dict[str, Any] = None, priority: int = 0,
                     deadline: Optional[float] = None, dependencies: List[str] = None) -> Task:
        task = Task(
            task_id="",
            title=title,
            description=description,
            type=task_type,
            input_data=input_data or {},
            priority=priority,
            deadline=deadline,
            dependencies=dependencies or []
        )
        
        self.tasks[task.task_id] = task
        self.pending_tasks.append(task.task_id)
        
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_pending_tasks(self) -> List[Task]:
        return [self.tasks[t] for t in self.pending_tasks]

    def get_in_progress_tasks(self) -> List[Task]:
        return [self.tasks[t] for t in self.in_progress_tasks]

    def get_completed_tasks(self) -> List[Task]:
        return [self.tasks[t] for t in self.completed_tasks]

    def bid_on_task(self, task_id: str, bidder_id: str,
                    bid_details: Dict[str, Any]) -> bool:
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        task.bids.append({
            "bidder_id": bidder_id,
            "timestamp": time.time(),
            **bid_details
        })
        task.updated_at = time.time()
        return True

    def assign_task(self, task_id: str, assignee_id: str) -> bool:
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        task.assignee = assignee_id
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = time.time()
        
        self.pending_tasks.remove(task_id)
        self.in_progress_tasks.append(task_id)
        
        return True

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if "output_data" in updates:
            task.output_data.update(updates["output_data"])
        if "status" in updates:
            task.status = updates["status"]
        if "priority" in updates:
            task.priority = updates["priority"]
        
        task.updated_at = time.time()
        
        if task.status == TaskStatus.COMPLETED and task_id in self.in_progress_tasks:
            self.in_progress_tasks.remove(task_id)
            self.completed_tasks.append(task_id)
        elif task.status == TaskStatus.FAILED and task_id in self.in_progress_tasks:
            self.in_progress_tasks.remove(task_id)
            task.retry_count += 1
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
                self.pending_tasks.append(task_id)
        
        return True

    def complete_task(self, task_id: str, output_data: Dict[str, Any]) -> bool:
        return self.update_task(task_id, {
            "status": TaskStatus.COMPLETED,
            "output_data": output_data
        })

    def fail_task(self, task_id: str, reason: str = "") -> bool:
        return self.update_task(task_id, {
            "status": TaskStatus.FAILED,
            "output_data": {"error": reason}
        })

    def cancel_task(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.updated_at = time.time()
        
        if task_id in self.pending_tasks:
            self.pending_tasks.remove(task_id)
        elif task_id in self.in_progress_tasks:
            self.in_progress_tasks.remove(task_id)
        
        return True

    def get_tasks_by_type(self, task_type: str) -> List[Task]:
        return [task for task in self.tasks.values() if task.type == task_type]

    def get_tasks_by_priority(self, priority: int) -> List[Task]:
        return [task for task in self.tasks.values() if task.priority == priority]

    def get_tasks_by_assignee(self, assignee_id: str) -> List[Task]:
        return [task for task in self.tasks.values() if task.assignee == assignee_id]

    def get_task_stats(self) -> Dict[str, int]:
        return {
            "total": len(self.tasks),
            "pending": len(self.pending_tasks),
            "in_progress": len(self.in_progress_tasks),
            "completed": len(self.completed_tasks),
            "failed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
            "cancelled": sum(1 for t in self.tasks.values() if t.status == TaskStatus.CANCELLED)
        }
    
    def get_task_type_stats(self) -> Dict[str, Dict[str, float]]:
        type_stats = {}
        
        for task in self.tasks.values():
            if task.type not in type_stats:
                type_stats[task.type] = {"completed": 0, "failed": 0, "success_rate": 0.0}
            
            if task.status == TaskStatus.COMPLETED:
                type_stats[task.type]["completed"] += 1
            elif task.status == TaskStatus.FAILED:
                type_stats[task.type]["failed"] += 1
        
        for task_type, stats in type_stats.items():
            total = stats["completed"] + stats["failed"]
            if total > 0:
                stats["success_rate"] = stats["completed"] / total
        
        return type_stats

    def cleanup_old_tasks(self, max_age_hours: float = 24):
        cutoff_time = time.time() - max_age_hours * 3600
        
        for task_id in list(self.tasks.keys()):
            task = self.tasks[task_id]
            if task.created_at < cutoff_time and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                self.tasks.pop(task_id)
                if task_id in self.completed_tasks:
                    self.completed_tasks.remove(task_id)