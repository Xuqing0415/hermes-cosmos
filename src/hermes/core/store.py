"""
In-memory implementations for development and testing
"""

import asyncio
import json
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional
from uuid import UUID

import structlog
from prometheus_client import Counter, Histogram, Gauge

from hermes.core.models import Job, JobStatus, JobPriority, Checkpoint, CheckpointState
from hermes.core.config import Region

logger = structlog.get_logger()

# Metrics
JOBS_SUBMITTED = Counter("hermes_jobs_submitted_total", "Total jobs submitted")
JOBS_COMPLETED = Counter("hermes_jobs_completed_total", "Total jobs completed", ["status"])
JOB_LATENCY = Histogram("hermes_job_latency_seconds", "Job scheduling latency")
CHECKPOINTS_CREATED = Counter("hermes_checkpoints_created_total", "Total checkpoints created")
ACTIVE_GPUS = Gauge("hermes_active_gpus", "Active GPUs", ["region"])
PENDING_JOBS = Gauge("hermes_pending_jobs", "Pending jobs in queue")


class InMemoryJobQueue:
    """In-memory job queue for development"""
    
    def __init__(self):
        self._jobs: Dict[UUID, Job] = {}
        self._queue: Deque[UUID] = deque()
        self._priority_queues: Dict[JobPriority, Deque[UUID]] = {
            JobPriority.CRITICAL: deque(),
            JobPriority.HIGH: deque(),
            JobPriority.NORMAL: deque(),
            JobPriority.LOW: deque(),
        }
        self._lock = asyncio.Lock()
    
    async def enqueue(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.id] = job
            self._priority_queues[job.priority].append(job.id)
            JOBS_SUBMITTED.inc()
            PENDING_JOBS.inc()
            logger.info("Job enqueued", job_id=str(job.id), priority=job.priority.value)
    
    async def dequeue(self) -> Optional[Job]:
        async with self._lock:
            for priority in [
                JobPriority.CRITICAL,
                JobPriority.HIGH,
                JobPriority.NORMAL,
                JobPriority.LOW,
            ]:
                if self._priority_queues[priority]:
                    job_id = self._priority_queues[priority].popleft()
                    PENDING_JOBS.dec()
                    return self._jobs.get(job_id)
            return None
    
    async def get_job(self, job_id: UUID) -> Optional[Job]:
        return self._jobs.get(job_id)
    
    async def update_job_status(self, job_id: UUID, status: JobStatus) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                job.updated_at = datetime.now(timezone.utc)
                if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    JOBS_COMPLETED.labels(status=status.value).inc()
                logger.info("Job status updated", job_id=str(job.id), status=status.value)
    
    async def get_all_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs


class InMemoryCheckpointStore:
    """In-memory checkpoint store for development"""
    
    def __init__(self, max_size: int = 1024 * 1024 * 1024):
        self._checkpoints: Dict[UUID, Checkpoint] = {}
        self._data: Dict[UUID, bytes] = {}
        self._max_size = max_size
        self._current_size = 0
        self._lock = asyncio.Lock()
    
    async def create(self, checkpoint: Checkpoint, data: bytes) -> Checkpoint:
        async with self._lock:
            # Reject data that exceeds the max store size
            if len(data) > self._max_size:
                raise ValueError(f"Data size {len(data)} exceeds max store size {self._max_size}")

            # Check if we need to evict old checkpoints
            while self._current_size + len(data) > self._max_size and self._checkpoints:
                oldest_id = min(self._checkpoints.keys(), key=lambda k: self._checkpoints[k].created_at)
                await self._delete_unsafe(oldest_id)
            
            self._checkpoints[checkpoint.id] = checkpoint
            self._data[checkpoint.id] = data
            self._current_size += len(data)
            CHECKPOINTS_CREATED.inc()
            logger.info("Checkpoint created", checkpoint_id=str(checkpoint.id), size=len(data))
            
            checkpoint.state = CheckpointState.COMPLETED
            checkpoint.size_bytes = len(data)
            return checkpoint
    
    async def get(self, checkpoint_id: UUID) -> Optional[Checkpoint]:
        return self._checkpoints.get(checkpoint_id)
    
    async def load_data(self, checkpoint_id: UUID) -> Optional[bytes]:
        return self._data.get(checkpoint_id)
    
    async def delete(self, checkpoint_id: UUID) -> bool:
        async with self._lock:
            return await self._delete_unsafe(checkpoint_id)
    
    async def _delete_unsafe(self, checkpoint_id: UUID) -> bool:
        if checkpoint_id not in self._checkpoints:
            return False
        
        checkpoint = self._checkpoints.pop(checkpoint_id)
        data = self._data.pop(checkpoint_id)
        self._current_size -= len(data)
        logger.info("Checkpoint deleted", checkpoint_id=str(checkpoint_id))
        return True
    
    async def list_by_job(self, job_id: UUID) -> List[Checkpoint]:
        return [cp for cp in self._checkpoints.values() if cp.job_id == job_id]


class InMemoryResourceManager:
    """In-memory resource manager for development"""
    
    def __init__(self):
        self._regions: Dict[Region, Dict[str, Any]] = {
            Region.US_EAST: {"total_gpus": 500, "available_gpus": 500, "nodes": []},
            Region.US_WEST: {"total_gpus": 500, "available_gpus": 500, "nodes": []},
            Region.EU_WEST: {"total_gpus": 500, "available_gpus": 500, "nodes": []},
        }
        self._allocated_gpus: Dict[UUID, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def get_cluster_state(self) -> Dict[str, Any]:
        regions_data = {}
        total_gpus = 0
        available_gpus = 0
        
        for region, data in self._regions.items():
            regions_data[region.value] = {
                "total": data["total_gpus"],
                "available": data["available_gpus"],
                "allocated": data["total_gpus"] - data["available_gpus"],
            }
            total_gpus += data["total_gpus"]
            available_gpus += data["available_gpus"]
        
        return {
            "regions": regions_data,
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "allocated_gpus": total_gpus - available_gpus,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    async def allocate(self, job: Job, region: Region, gpu_count: int) -> bool:
        async with self._lock:
            if region not in self._regions:
                return False
            
            region_data = self._regions[region]
            if region_data["available_gpus"] < gpu_count:
                return False
            
            region_data["available_gpus"] -= gpu_count
            self._allocated_gpus[job.id] = {
                "region": region,
                "gpu_count": gpu_count,
                "allocated_at": datetime.now(timezone.utc),
            }
            ACTIVE_GPUS.labels(region=region.value).set(
                region_data["total_gpus"] - region_data["available_gpus"]
            )
            logger.info("Resources allocated", job_id=str(job.id), region=region.value, gpu_count=gpu_count)
            return True
    
    async def release(self, job_id: UUID) -> bool:
        async with self._lock:
            if job_id not in self._allocated_gpus:
                return False
            
            allocation = self._allocated_gpus.pop(job_id)
            region = allocation["region"]
            gpu_count = allocation["gpu_count"]
            
            self._regions[region]["available_gpus"] += gpu_count
            ACTIVE_GPUS.labels(region=region.value).set(
                self._regions[region]["total_gpus"] - self._regions[region]["available_gpus"]
            )
            logger.info("Resources released", job_id=str(job_id), region=region.value, gpu_count=gpu_count)
            return True
    
    async def find_available_region(
        self,
        gpu_count: int,
        preferred_regions: Optional[List[Region]] = None,
        carbon_aware: bool = True,
    ) -> Optional[Region]:
        regions = preferred_regions or list(self._regions.keys())
        
        for region in regions:
            if self._regions[region]["available_gpus"] >= gpu_count:
                return region
        
        # If no preferred region found, try any available
        for region in self._regions.keys():
            if self._regions[region]["available_gpus"] >= gpu_count:
                return region
        
        return None


# Singleton instances
_job_queue = None
_checkpoint_store = None
_resource_manager = None

def get_job_queue() -> InMemoryJobQueue:
    global _job_queue
    if _job_queue is None:
        _job_queue = InMemoryJobQueue()
    return _job_queue

def get_checkpoint_store() -> InMemoryCheckpointStore:
    global _checkpoint_store
    if _checkpoint_store is None:
        _checkpoint_store = InMemoryCheckpointStore()
    return _checkpoint_store

def get_resource_manager() -> InMemoryResourceManager:
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = InMemoryResourceManager()
    return _resource_manager
